# =====================================================================
#  BACK del chatbot de la tienda de Laura: una función Lambda en Python
#  Recibe la charla que manda el FRONT (por la Function URL) y le pregunta
#  al modelo en Amazon Bedrock.
#
#  El modelo NO se entrena aquí. Se le dan dos textos antes de cada charla:
#    instrucciones.txt → el SYSTEM PROMPT: quién es, cómo habla y qué no hace
#    catalogo.txt      → lo que sabe de la tienda (va dentro del system prompt)
#  Viven en el BACK, nunca en el front (allá cualquiera los cambia con F12).
#  Cambia cualquiera de los dos por pull request: al unirlo, el pipeline lo despliega.
# =====================================================================
import base64, json, os, urllib.request

import boto3
from botocore.exceptions import ClientError

# Esto corre UNA sola vez, cuando AWS prende una instancia nueva (arranque en frío)
bedrock = boto3.client("bedrock-runtime")
AQUI = os.path.dirname(os.path.abspath(__file__))


def leer(nombre):
    """Lee un texto del paquete. Aguanta archivos guardados desde Windows (con BOM o en ANSI)."""
    with open(os.path.join(AQUI, nombre), "rb") as archivo:
        crudo = archivo.read()
    try:
        return crudo.decode("utf-8-sig")
    except UnicodeDecodeError:
        return crudo.decode("cp1252", errors="replace")


SISTEMA = leer("instrucciones.txt") + "\n\n" + leer("catalogo.txt")

MODELO = os.environ["MODELO"]  # lo pone Terraform (variable "modelo")
TEMPERATURA = float(os.environ.get("TEMPERATURA", "0.3"))
MAX_TOKENS = 300  # el tope de la respuesta: también es el tope de lo que cuesta
MAX_MENSAJES = 12  # la mochila del contexto: solo viajan los últimos 12 mensajes
MAX_LETRAS = 600  # un mensaje del cliente no puede ser más largo que esto

# El RESPALDO DEL PROFE: si TU cuenta no tiene cuota de Bedrock (ThrottlingException),
# la pregunta viaja a la Lambda del profe, con TUS instrucciones y TU catálogo.
# Vacío = sin respaldo. Lo pone Terraform desde el secreto RESPALDO_URL del repositorio.
RESPALDO = os.environ.get("RESPALDO", "")


def responder(codigo, cuerpo, tipo="application/json"):
    if tipo == "application/json":
        cuerpo = json.dumps(cuerpo, ensure_ascii=False)
    return {"statusCode": codigo, "headers": {"content-type": tipo + "; charset=utf-8"}, "body": cuerpo}


def limpiar(mensajes):
    """Deja solo lo que el modelo acepta: turnos user/assistant con texto, empezando por user."""
    limpios = []
    for m in mensajes[-MAX_MENSAJES:]:
        rol, texto = m.get("rol"), str(m.get("texto", "")).strip()[:MAX_LETRAS]
        if rol not in ("user", "assistant") or not texto:
            continue
        if limpios and limpios[-1]["role"] == rol:  # dos turnos seguidos del mismo: se queda el último
            limpios.pop()
        limpios.append({"role": rol, "content": [{"text": texto}]})
    while limpios and limpios[0]["role"] != "user":
        limpios.pop(0)
    return limpios


def preguntar_al_respaldo(mensajes, cuenta):
    datos = json.dumps({"sistema": SISTEMA, "mensajes": mensajes, "temperatura": TEMPERATURA, "cuenta": cuenta})
    pedido = urllib.request.Request(RESPALDO, data=datos.encode("utf-8"), headers={"content-type": "application/json"})
    with urllib.request.urlopen(pedido, timeout=20) as r:
        return json.load(r)


def handler(event, context):
    metodo = event.get("requestContext", {}).get("http", {}).get("method", "POST")

    if metodo != "POST":
        return responder(405, {"error": "el chat solo recibe POST"})

    try:
        cuerpo = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            cuerpo = base64.b64decode(cuerpo).decode("utf-8")
        mensajes = limpiar(json.loads(cuerpo).get("mensajes", []))
    except (ValueError, AttributeError):
        return responder(400, {"error": "la conversación no llegó en JSON"})
    if not mensajes or mensajes[-1]["role"] != "user":
        return responder(400, {"error": "falta la pregunta del cliente"})

    try:
        r = bedrock.converse(
            modelId=MODELO,
            system=[{"text": SISTEMA}],
            messages=mensajes,
            inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": TEMPERATURA},
        )
    except ClientError as e:
        # AccessDeniedException = al rol le falta el permiso
        # ThrottlingException   = se acabó la cuota de Bedrock de TU cuenta (por minuto o por día)
        codigo = e.response["Error"]["Code"]
        print(json.dumps({"evento": "error", "codigo": codigo, "detalle": e.response["Error"]["Message"]}))
        if codigo == "ThrottlingException" and RESPALDO:
            try:
                j = preguntar_al_respaldo(mensajes, context.invoked_function_arn.split(":")[4])
                print(json.dumps({"evento": "respaldo", "entrada": j["tokens"]["entrada"], "salida": j["tokens"]["salida"]}))
                return responder(200, dict(j, temperatura=TEMPERATURA, respaldo=True))
            except Exception as falla:  # el respaldo tampoco respondió: se muestra el error original
                print(json.dumps({"evento": "respaldo-fallo", "detalle": str(falla)[:200]}))
        return responder(502, {"error": codigo, "detalle": e.response["Error"]["Message"]})

    texto = r["output"]["message"]["content"][0]["text"]
    uso = r["usage"]
    # Una línea en JSON por respuesta: se puede preguntar con Logs Insights
    print(json.dumps({"evento": "respuesta", "modelo": MODELO, "entrada": uso["inputTokens"],
                      "salida": uso["outputTokens"], "ms": r["metrics"]["latencyMs"]}))
    return responder(200, {
        "texto": texto,
        "modelo": MODELO,
        "temperatura": TEMPERATURA,
        "tokens": {"entrada": uso["inputTokens"], "salida": uso["outputTokens"]},
        "ms": r["metrics"]["latencyMs"],
    })
