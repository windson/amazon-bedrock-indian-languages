{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "24694bff-a63c-463c-832b-588c653f128f",
   "metadata": {},
   "source": [
    "# Amazon Bedrock Claude usecases for Indian Languages\n",
    "\n",
    "*This notebook should work well with the **`Data Science 3.0`** kernel in SageMaker Studio*\n",
    "\n",
    "Contents\n",
    "- Claude Model Selection    \n",
    "- Hindi Examples\n",
    "    - Question Answering\n",
    "    - Summarization\n",
    "    - Translate\n",
    "    - Transliterate\n",
    "- Telugu Examples\n",
    "    - Question Answering\n",
    "    - Summarization\n",
    "    - Translate\n",
    "    - Transliterate"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d936104f-513e-4d66-827a-1f6b0ba8d30a",
   "metadata": {},
   "source": [
    "---\n",
    "## Claude Model Selection\n",
    "\n",
    "We have following Claude 3 models available as of today on Amazon Bedrock\n",
    "\n",
    "| Model Name | Model ID |\n",
    "|--|--|\n",
    "| Claude 3 Haiku | anthropic.claude-3-haiku-20240307-v1:0 |\n",
    "| Claude 3 Sonnet | anthropic.claude-3-sonnet-20240229-v1:0 |\n",
    "\n",
    "- [Anthropic Claude Messages API](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html#model-parameters-anthropic-claude-messages-overview)\n",
    "- [Create a Message](https://docs.anthropic.com/claude/reference/messages_post)\n",
    "- [Amazon Bedrock API for Claude](https://docs.anthropic.com/claude/reference/claude-on-amazon-bedrock)\n",
    "- [Amazon Bedrock Pricing Page](https://aws.amazon.com/bedrock/pricing/)\n",
    "---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "08945134-9047-426d-944d-aeb8e0c9e518",
   "metadata": {},
   "outputs": [],
   "source": [
    "claude3_haiku_id = \"anthropic.claude-3-haiku-20240307-v1:0\"\n",
    "claude3_sonnet_id = \"anthropic.claude-3-sonnet-20240229-v1:0\"\n",
    "\n",
    "DEFAULT_MODEL = claude3_sonnet_id"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "10a329e3-0deb-4688-aca5-c9e507f5f879",
   "metadata": {},
   "outputs": [],
   "source": [
    "import boto3\n",
    "import json\n",
    "\n",
    "class LLM:\n",
    "    def __init__(self, model_id):\n",
    "        self.model_id = model_id\n",
    "        self.bedrock = boto3.client(service_name=\"bedrock-runtime\")\n",
    "        \n",
    "    def invoke(self, system, messages, max_tokens=128):\n",
    "        body = json.dumps({\n",
    "                  \"max_tokens\": max_tokens,\n",
    "                  \"system\": system,\n",
    "                  \"messages\": messages,\n",
    "                  \"anthropic_version\": \"bedrock-2023-05-31\"\n",
    "                })\n",
    "        response = self.bedrock.invoke_model(\n",
    "                                            body=body, \n",
    "                                            modelId=self.model_id)\n",
    "\n",
    "        response_body = json.loads(response.get(\"body\").read())\n",
    "        return response_body['content']\n",
    "    \n",
    "llm = LLM(DEFAULT_MODEL)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "cb988a84-298e-42da-90f7-8f94355224d5",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "def system_prompt(language, task= 'qna'):\n",
    "    sys_prompt = f\"Hello, You are an expert in {language}, an Indian language.\"\n",
    "    if task == 'qna':\n",
    "        sys_prompt = sys_prompt + f\"\"\"\\nGiven the content in {language} wrapped between <{language}> and </{language}> tags, there will be follow up questions asked to you in English between <question> and </question> tags. You respond in {language} language. You respond to the point and not verbose. Your response will be between the tags <response> and </response>. For responses that requires to identify entities, you just include those entities and nothing more verbose in the response tags.\n",
    "          You are not verbose and your response doesn't include response out of context given to you. \n",
    "          If the question doesn't have a relevant answer in the givent content, you must answer \"I don't have an answer\" translated in {language} language.\n",
    "          \"\"\"\n",
    "    elif task == 'summarization':\n",
    "        sys_prompt = f\"\"\"\\nGiven the content in {language} wrapped between <{language}> and </{language}> tags, summarize the content. \n",
    "          You respond in {language} language. You summarize to the point and not verbose.\n",
    "          You are not verbose and your response doesn't include response out of context given to you. \n",
    "          If the content is not in {language}, you must answer \"Please provide content in {language} language\" translated in {language} language.\n",
    "          \"\"\"\n",
    "\n",
    "    return sys_prompt\n",
    "\n",
    "def trans_system_prompt(source, target, task):\n",
    "    doer = 'transliterator' if task == 'transliterate' else 'translator'\n",
    "    sys_prompt = f\"Hello, You are an expert {doer}. For content provided in {source}, language perform {task} task to {target} language.\"\n",
    "    sys_prompt = sys_prompt + f\"\"\"\\n Given the content provided in {source} language wrapped between <{source}> and </{source}> tags, {task} the content to {target} language. \n",
    "    You are not verbose and your response doesn't include response out of context given to you. \n",
    "    If you don't know the {target} language or cannot perform {task}, you must answer \"I don't have an answer\" translated in {target} language.\n",
    "    \"\"\"\n",
    "    return sys_prompt\n",
    "\n",
    "def task_prompt_builder(content, language, task, tasktag):\n",
    "    actual_question = [\n",
    "                        {\"role\": \"user\", \"content\": f\"\"\"<{language}>{content}</{language}>\n",
    "                        <{tasktag}>{task}</{tasktag}>\"\"\"}\n",
    "                    ]\n",
    "    return actual_question\n",
    "\n",
    "def ask_question(llm, language, language_examples, content, question):\n",
    "    actual_question = task_prompt_builder(content, language, question, 'question')\n",
    "    message = language_examples + actual_question\n",
    "    response_text = llm.invoke(\n",
    "        system_prompt(language),\n",
    "        message,\n",
    "        max_tokens=1500,\n",
    "    )\n",
    "    if response_text:\n",
    "        print(response_text[0].get('text'))\n",
    "\n",
    "def summarize(llm, language, content):\n",
    "    task = f\"Summarize the content in {language}. Your response will be between the tags <summary> and </summary>.\"\n",
    "    message = task_prompt_builder(content, language, task, 'summary') \n",
    "    response_text = llm.invoke(\n",
    "        system_prompt(language, 'summarize'),\n",
    "        message,\n",
    "        max_tokens=1500,\n",
    "    )\n",
    "    if response_text:\n",
    "        print(response_text[0].get('text'))\n",
    "        \n",
    "def transliterate(llm, source, target, content):\n",
    "    task = f\"Transliterate the content in {source} language to {target} language. Wrap the response between <{target}Transliteration> and </{target}Transliteration> tags.\"\n",
    "    message = task_prompt_builder(content, source, task, 'transliterate') \n",
    "    response_text = llm.invoke(\n",
    "        trans_system_prompt(source, target, 'transliterate'),\n",
    "        message,\n",
    "        max_tokens=1500,\n",
    "    )\n",
    "    if response_text:\n",
    "        print(response_text[0].get('text'))\n",
    "        \n",
    "def translate(llm, source, target, content):\n",
    "    task = f\"Translate the content in {source} language to {target} language. Wrap the response between <{target}Translation> and </{target}Translation> tags.\"\n",
    "    message = task_prompt_builder(content, source, task, 'translate') \n",
    "    response_text = llm.invoke(\n",
    "        trans_system_prompt(source, target, 'translate'),\n",
    "        message,\n",
    "        max_tokens=1500,\n",
    "    )\n",
    "    if response_text:\n",
    "        print(response_text[0].get('text'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f2992193-7473-4fe8-bd6f-ef622430c7fc",
   "metadata": {},
   "source": [
    "## Hindi Examples\n",
    "### Question Answering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "54c92529-d4d1-4674-bde2-2ce0567a2b43",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "language = 'Hindi'\n",
    "# few shot examples\n",
    "hindi_examples = [\n",
    "    {\"role\": \"user\", \"content\": f\"\"\"<{language}>लखनऊ के प्रसिद्ध चंद्रिका देवी मंदिर की महिमा और इतिहास बेहद अद्भुत है. मंदिर रामायण और महाभारत काल से जुड़ा हुआ है और यहाँ की वर्तमान शुरुआत 300 साल पुरानी है. देशभर से भक्त यहाँ अपनी मनोकामना लेकर आते हैं और देवी की शक्ति को नमन करते हैं. देखें वीडियो.</{language}>\n",
    "    <question>Any location info present in the news?</question>\"\"\"},\n",
    "    {\"role\": \"assistant\", \"content\": \"<response>लखनऊ</response>\"},\n",
    "    {\"role\": \"user\", \"content\": f\"\"\"<question>What is famous in this news?</question>\"\"\"},\n",
    "    {\"role\": \"assistant\", \"content\": \"<response>चंद्रिका देवी मंदिर की महिमा और इतिहास</response>\"},\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1025fd03-e550-49c6-9f2f-e9f4c30f347c",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "content = \"\"\"सारा अली खान स्टारर 'ऐ वतन मेरे वतन' अमेजन प्राइम पर स्ट्रीम हुई है. जो कि एक बायोपिक है. फिल्म की कहानी क्रांतिकारी उषा मेहता के जीवन पर आधारित है,  जिन्हें एक सच्ची गांधीवादी और कांग्रेस रेडियो की फाउंडर के रूप में भी जाना जाता है. बताते हैं कैसी है ये फिल्म और किरदार को जीवंत करने में सारा अली खान कितनी खरी उतर पाईं.\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "787df196-af79-4386-b5b5-5ce36e7b5fb3",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<response>यह सारा अली खान अभिनीत फिल्म 'ऐ वतन मेरे वतन' के बारे में है, जो क्रांतिकारी उषा मेहता की बायोपिक है।</response>\n"
     ]
    }
   ],
   "source": [
    "question = \"What is the content about?\"\n",
    "ask_question(llm, language, hindi_examples, content, question)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "e0d04c11-fa4d-417f-ae75-74fae7c5c392",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<response>उषा मेहता</response>\n"
     ]
    }
   ],
   "source": [
    "question = \"Who is the person being discussed about?\"\n",
    "ask_question(llm,language, hindi_examples, content, question)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "f6ca0df6-bd80-4297-848e-0e9e429a04b8",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<response>सारा अली खान</response>\n"
     ]
    }
   ],
   "source": [
    "question = \"Who is the actor being discussed about?\"        \n",
    "ask_question(llm,language, hindi_examples, content, question)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "41b49cb0-aca6-4385-8f3f-ed91c9d9128b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<response>\n",
      "{'सारा अली खान': 'person', 'उषा मेहता': 'person', 'ऐ वतन मेरे वतन': 'moviename', 'अमेजन प्राइम': 'place', 'कांग्रेस रेडियो': 'celebrity'}\n",
      "</response>\n"
     ]
    }
   ],
   "source": [
    "question = \"\"\"\n",
    "list all the named entities and their nature such as person, moviename, place, city, type etc in the format \n",
    "{'ent1': 'person', 'ent2': 'person', 'ent2': 'city','ent4': 'place', 'ent5':'celebrity'}\n",
    "\"\"\"\n",
    "ask_question(llm,language, hindi_examples, content, question)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "99b04324-6636-4648-9f08-b26cb16397f2",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<response>मैं नहीं देखता कि इस प्रदत्त सामग्री में किसी जानवर का उल्लेख है।</response>\n"
     ]
    }
   ],
   "source": [
    "question = \"Which animal is mentioned?\"        \n",
    "ask_question(llm,language, hindi_examples, content, question)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a388dab1-4cd5-473e-a903-c8800e57d44a",
   "metadata": {},
   "source": [
    "### Summarization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "dfdf5f09-923d-4c7d-907a-106690afb812",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<summary>\n",
      "यह राम नवमी बहुत खास है क्योंकि इस बार भक्त अपने आराध्य भगवान राम का दर्शन अयोध्या में बने नए भव्य राम मंदिर में कर सकेंगे। दोपहर 12 बजे बालक राम का सूर्याभिषेक किया जाएगा जो बहुत ही शुभ मुहूर्त है। यह पहली बार है जब सदियों बाद भक्तों को राम मंदिर में दर्शन करने का अवसर मिलेगा।\n",
      "</summary>\n"
     ]
    }
   ],
   "source": [
    "content = \"\"\"\n",
    "इस बार राम नवमी न केवल अयोध्या के लिए बल्कि दुनिया भर के राम भक्तों के लिए बेहद खास है. \n",
    "सदियों बाद यह पहली राम नवमी जब भक्त अपने आराध्य का दर्शन भव्य राम मंदिर में दोपहर 12 बजे भगवान राम के बाल स्वरूप का सूर्य तिलक होगा. \n",
    "शुभ मुहूर्त में बालक राम का सूर्याभिषेक किया जाएगा. राम नवमी पर सूर्य तिलक के लिए सुबह 11 बजकर 05 मिनट से दोपहर 1 बजकर 35 मिनट तक शुभ मुहूर्त रहेगा.\n",
    "\"\"\"\n",
    "summarize(llm, language, content)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "34da8837-bf3e-4d4b-ad74-b2ef3ab72b87",
   "metadata": {
    "tags": []
   },
   "source": [
    "### Translate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "148591af-988b-4f24-9a96-04f1fe9f1115",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<MarathiTranslation>\n",
      "या वेळी रामनवमी फक्त अयोध्येसाठीच नव्हे तर जगभरातील रामभक्तांसाठी खूप विशेष आहे.\n",
      "शेकडो वर्षांनंतर ही पहिली रामनवमी आहे जेव्हा भक्त त्यांच्या आराध्य देवतेचे दर्शन भव्य राममंदिरात दुपारी 12 वाजता भगवान रामाच्या बालस्वरूपाच्या सूर्यतिलकाचे करू शकतील.\n",
      "शुभ मुहूर्तात बालक रामाचे सूर्याभिषेक केले जाईल. रामनवमीच्या दिवशी सूर्यतिलकासाठी सकाळी 11.05 ते दुपारी 1.35 या काळात शुभ मुहूर्त राहील.\n",
      "</MarathiTranslation>\n"
     ]
    }
   ],
   "source": [
    "target_language = 'Marathi'\n",
    "translate(llm, language, target_language, content)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f24c983b-a24e-470b-b4e5-f9e57b115d26",
   "metadata": {},
   "source": [
    "### Transliterate\n",
    "\n",
    "Note: `anthropic.claude-3-sonnet-20240229-v1:0` is better at transliteration."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "a24e3050-7ab3-43db-bf25-2400a7110c42",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<EnglishTransliteration>\n",
      "Is baar Raam Navmi na kevala Ayodhya ke lie balki duniya bhar ke Raam bhakton ke lie behad khaas hai.\n",
      "Sadiyon baad yeh pehli Raam Navmi jab bhakt apne aaradhya ka darshan bhavya Raam Mandir mein dopahar 12 baje Bhagavaan Raam ke baal swaroop ka soory tilak hoga.\n",
      "Shubh muhoort mein baalak Raam ka sooryaabhishek kiya jaayega. Raam Navmi par soory tilak ke lie subah 11 bajekar 05 minute se dopahar 1 bajekar 35 minute tak shubh muhoort rahega.\n",
      "</EnglishTransliteration>\n"
     ]
    }
   ],
   "source": [
    "target_language = 'English'\n",
    "transliterate(llm, language, target_language, content)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "457535bf-d88e-4ab5-bc66-c4eba0ddc16e",
   "metadata": {},
   "source": [
    "## Telugu Examples\n",
    "### Question Answering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "f29398de-0ffd-4596-9b01-99d81831b0fc",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "language = 'Telugu'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "2fa72578-e56e-47ac-bdb1-3b7a0cd0c3bc",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "content = \"\"\"\n",
    "ఎన్‌పీసీఐకు సంబంధించిన తాజా ప్రకటనలో క్రెడిట్ కార్డు ద్వారా ఈఎంఐ సదుపాయం, క్రెడిట్ ఖాతా బిల్లు చెల్లింపు, వాయిదా చెల్లింపు ఎంపికలు, పరిమితి నిర్వహణ కార్యాచరణల వంటి అనేక కీలక సదుపాయాలను అందిస్తున్నాయి. \n",
    "మే 31, 2024లోపు ఈ ఫీచర్లను అమలు చేయాలని బ్యాంకులు, కార్డు జారీచేసే వారితో సహా జారీ చేసే సంస్థలు ఆదేశించింది.\n",
    "జూన్ 30 2024 లోగా అన్ని వివరాలు ఆధార్ కార్డ్ లొ పొందు పరచాలి.\n",
    "\"\"\"\n",
    "telugu_examples = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "47e713b7-f0fd-4844-986e-73ed4c30d784",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<response>ఈ విషయం క్రెడిట్ కార్డ్‌లకు సంబంధించిన కొత్త సదుపాయాలు, ఫీచర్లపై మాట్లాడుతుంది మరియు ఈ కొత్త కార్యాచరణలను అమలు చేయడానికి గడువు వరకు పేర్కొంది</response>\n"
     ]
    }
   ],
   "source": [
    "question = \"What is the content about?\"\n",
    "ask_question(llm,language, telugu_examples, content, question)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "564a5c0c-c969-4f14-8e30-8e7d8a095d67",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<response>సార్వజనిక బ్యాంకులు, కార్డు జారీచేసే వారు</response>\n"
     ]
    }
   ],
   "source": [
    "question = \"Who is the person being discussed about?\"\n",
    "ask_question(llm,language, telugu_examples, content, question)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "1daae131-db71-4497-b205-67816cc266ed",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<response>నేను ఈ సమాచారంలో నటుడి గురించి ఏమీ చూడలేదు</response>\n"
     ]
    }
   ],
   "source": [
    "question = \"Who is the actor being discussed about?\"        \n",
    "ask_question(llm,language, telugu_examples, content, question)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "4f73002e-90e1-4c7c-8d38-a54aaa15aafc",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<response>అవును, ప్రకటనలో క్రెడిట్ కార్డు ద్వారా ఈఎంఐ సదుపాయం, క్రెడిట్ ఖాతా బిల్లు చెల్లింపు, వాయిదా చెల్లింపు ఎంపికలు, పరిమితి నిర్వహణ కార్యాచరణల వంటి అనేక కీలక సదుపాయాలను అందిస్తున్నట్లు తెలుపబడింది</response>\n"
     ]
    }
   ],
   "source": [
    "question = \"Is there any benefit described in news?\"        \n",
    "ask_question(llm,language, telugu_examples, content, question)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "d71cbd33-dff0-44fa-9be6-299000877393",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<response>\n",
      "{'31-05-2024': ['క్రెడిట్ కార్డు ద్వారా ఈఎంఐ సదుపాయం, క్రెడిట్ ఖాతా బిల్లు చెల్లింపు, వాయిదా చెల్లింపు ఎంపికలు, పరిమితి నిర్వహణ కార్యాచరణలు అమలు చేయాలి'],\n",
      " '30-06-2024': ['అన్ని వివరాలు ఆధార్ కార్డ్ లొ పొందు పరచాలి']}\n",
      "</response>\n"
     ]
    }
   ],
   "source": [
    "question = \"Any dates or deadlines? respond with {'DD-MM-YYYY': [action1, action2, ..], 'DD-MM-YYY': [action1]}\"        \n",
    "ask_question(llm,language, telugu_examples, content, question)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ffa084ee-f3e3-49e5-ad66-4cc7d7134de2",
   "metadata": {},
   "source": [
    "### Summarization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "cc9cc021-466f-4676-8e20-10c3b5bb6df6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<summary>\n",
      "ఎన్పీసీఐ క్రెడిట్ కార్డుల కోసం క్రెడిట్ ఖాతా చెల్లింపు, వాయిదా చెల్లింపు, ఈఎంఐ సదుపాయం, పరిమితి నిర్వహణ వంటి కీలక సదుపాయాలను పరిచయం చేసింది. బ్యాంకులు, కార్డు జారీ చేసే సంస్థలు మే 31, 2024 నాటికి ఈ ఫీచర్లను అమలు చేయాలని, జూన్ 30, 2024 నాటికి ఆధార్ కార్డులో అన్ని వివరాలను పొందుపరచాలని ఆదేశించింది.\n",
      "</summary>\n"
     ]
    }
   ],
   "source": [
    "summarize(llm, language, content)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "def62797-9c97-4580-a121-b75f6a48ba75",
   "metadata": {},
   "source": [
    "### Translate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "a33087fa-dcf4-432f-a736-a7df1b6a861e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<EnglishTranslation>Krishna planted flower trees for his friends.</EnglishTranslation>\n"
     ]
    }
   ],
   "source": [
    "content = \"కృష్ణుడు తన స్నేహితుల కోసం పూలవృక్షాలను నాటాడు.\"\n",
    "target_language = 'English'\n",
    "translate(llm, language, target_language, content)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "efb9b8b4-79e9-40b3-a620-fd484ed9afcd",
   "metadata": {
    "tags": []
   },
   "source": [
    "### Transliteration\n",
    "\n",
    "Transliteration is the process of representing or intending to represent a word, phrase, or text in a different script or writing system"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "353bc224-7180-4087-88f8-4c6bfebf2aa1",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<EnglishTransliteration>Krishna nATADu tana snehitula kosam pUlavrkshAlanu nATADu.</EnglishTransliteration>\n"
     ]
    }
   ],
   "source": [
    "target_language = 'English'\n",
    "transliterate(llm, language, target_language, content)"
   ]
  }
 ],
 "metadata": {
  "availableInstances": [
   {
    "_defaultOrder": 0,
    "_isFastLaunch": true,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 4,
    "name": "ml.t3.medium",
    "vcpuNum": 2
   },
   {
    "_defaultOrder": 1,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 8,
    "name": "ml.t3.large",
    "vcpuNum": 2
   },
   {
    "_defaultOrder": 2,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 16,
    "name": "ml.t3.xlarge",
    "vcpuNum": 4
   },
   {
    "_defaultOrder": 3,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.t3.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 4,
    "_isFastLaunch": true,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 8,
    "name": "ml.m5.large",
    "vcpuNum": 2
   },
   {
    "_defaultOrder": 5,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 16,
    "name": "ml.m5.xlarge",
    "vcpuNum": 4
   },
   {
    "_defaultOrder": 6,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.m5.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 7,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 64,
    "name": "ml.m5.4xlarge",
    "vcpuNum": 16
   },
   {
    "_defaultOrder": 8,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 128,
    "name": "ml.m5.8xlarge",
    "vcpuNum": 32
   },
   {
    "_defaultOrder": 9,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 192,
    "name": "ml.m5.12xlarge",
    "vcpuNum": 48
   },
   {
    "_defaultOrder": 10,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 256,
    "name": "ml.m5.16xlarge",
    "vcpuNum": 64
   },
   {
    "_defaultOrder": 11,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 384,
    "name": "ml.m5.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 12,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 8,
    "name": "ml.m5d.large",
    "vcpuNum": 2
   },
   {
    "_defaultOrder": 13,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 16,
    "name": "ml.m5d.xlarge",
    "vcpuNum": 4
   },
   {
    "_defaultOrder": 14,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.m5d.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 15,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 64,
    "name": "ml.m5d.4xlarge",
    "vcpuNum": 16
   },
   {
    "_defaultOrder": 16,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 128,
    "name": "ml.m5d.8xlarge",
    "vcpuNum": 32
   },
   {
    "_defaultOrder": 17,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 192,
    "name": "ml.m5d.12xlarge",
    "vcpuNum": 48
   },
   {
    "_defaultOrder": 18,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 256,
    "name": "ml.m5d.16xlarge",
    "vcpuNum": 64
   },
   {
    "_defaultOrder": 19,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 384,
    "name": "ml.m5d.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 20,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": true,
    "memoryGiB": 0,
    "name": "ml.geospatial.interactive",
    "supportedImageNames": [
     "sagemaker-geospatial-v1-0"
    ],
    "vcpuNum": 0
   },
   {
    "_defaultOrder": 21,
    "_isFastLaunch": true,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 4,
    "name": "ml.c5.large",
    "vcpuNum": 2
   },
   {
    "_defaultOrder": 22,
    "_isFastLaunch": false,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 8,
    "name": "ml.c5.xlarge",
    "vcpuNum": 4
   },
   {
    "_defaultOrder": 23,
    "_isFastLaunch": false,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 16,
    "name": "ml.c5.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 24,
    "_isFastLaunch": false,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.c5.4xlarge",
    "vcpuNum": 16
   },
   {
    "_defaultOrder": 25,
    "_isFastLaunch": false,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 72,
    "name": "ml.c5.9xlarge",
    "vcpuNum": 36
   },
   {
    "_defaultOrder": 26,
    "_isFastLaunch": false,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 96,
    "name": "ml.c5.12xlarge",
    "vcpuNum": 48
   },
   {
    "_defaultOrder": 27,
    "_isFastLaunch": false,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 144,
    "name": "ml.c5.18xlarge",
    "vcpuNum": 72
   },
   {
    "_defaultOrder": 28,
    "_isFastLaunch": false,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 192,
    "name": "ml.c5.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 29,
    "_isFastLaunch": true,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 16,
    "name": "ml.g4dn.xlarge",
    "vcpuNum": 4
   },
   {
    "_defaultOrder": 30,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.g4dn.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 31,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 64,
    "name": "ml.g4dn.4xlarge",
    "vcpuNum": 16
   },
   {
    "_defaultOrder": 32,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 128,
    "name": "ml.g4dn.8xlarge",
    "vcpuNum": 32
   },
   {
    "_defaultOrder": 33,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 4,
    "hideHardwareSpecs": false,
    "memoryGiB": 192,
    "name": "ml.g4dn.12xlarge",
    "vcpuNum": 48
   },
   {
    "_defaultOrder": 34,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 256,
    "name": "ml.g4dn.16xlarge",
    "vcpuNum": 64
   },
   {
    "_defaultOrder": 35,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 61,
    "name": "ml.p3.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 36,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 4,
    "hideHardwareSpecs": false,
    "memoryGiB": 244,
    "name": "ml.p3.8xlarge",
    "vcpuNum": 32
   },
   {
    "_defaultOrder": 37,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 8,
    "hideHardwareSpecs": false,
    "memoryGiB": 488,
    "name": "ml.p3.16xlarge",
    "vcpuNum": 64
   },
   {
    "_defaultOrder": 38,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 8,
    "hideHardwareSpecs": false,
    "memoryGiB": 768,
    "name": "ml.p3dn.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 39,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 16,
    "name": "ml.r5.large",
    "vcpuNum": 2
   },
   {
    "_defaultOrder": 40,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.r5.xlarge",
    "vcpuNum": 4
   },
   {
    "_defaultOrder": 41,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 64,
    "name": "ml.r5.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 42,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 128,
    "name": "ml.r5.4xlarge",
    "vcpuNum": 16
   },
   {
    "_defaultOrder": 43,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 256,
    "name": "ml.r5.8xlarge",
    "vcpuNum": 32
   },
   {
    "_defaultOrder": 44,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 384,
    "name": "ml.r5.12xlarge",
    "vcpuNum": 48
   },
   {
    "_defaultOrder": 45,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 512,
    "name": "ml.r5.16xlarge",
    "vcpuNum": 64
   },
   {
    "_defaultOrder": 46,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 768,
    "name": "ml.r5.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 47,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 16,
    "name": "ml.g5.xlarge",
    "vcpuNum": 4
   },
   {
    "_defaultOrder": 48,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.g5.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 49,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 64,
    "name": "ml.g5.4xlarge",
    "vcpuNum": 16
   },
   {
    "_defaultOrder": 50,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 128,
    "name": "ml.g5.8xlarge",
    "vcpuNum": 32
   },
   {
    "_defaultOrder": 51,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 256,
    "name": "ml.g5.16xlarge",
    "vcpuNum": 64
   },
   {
    "_defaultOrder": 52,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 4,
    "hideHardwareSpecs": false,
    "memoryGiB": 192,
    "name": "ml.g5.12xlarge",
    "vcpuNum": 48
   },
   {
    "_defaultOrder": 53,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 4,
    "hideHardwareSpecs": false,
    "memoryGiB": 384,
    "name": "ml.g5.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 54,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 8,
    "hideHardwareSpecs": false,
    "memoryGiB": 768,
    "name": "ml.g5.48xlarge",
    "vcpuNum": 192
   },
   {
    "_defaultOrder": 55,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 8,
    "hideHardwareSpecs": false,
    "memoryGiB": 1152,
    "name": "ml.p4d.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 56,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 8,
    "hideHardwareSpecs": false,
    "memoryGiB": 1152,
    "name": "ml.p4de.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 57,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.trn1.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 58,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 512,
    "name": "ml.trn1.32xlarge",
    "vcpuNum": 128
   },
   {
    "_defaultOrder": 59,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 512,
    "name": "ml.trn1n.32xlarge",
    "vcpuNum": 128
   }
  ],
  "instance_type": "ml.r5.4xlarge",
  "kernelspec": {
   "display_name": "Python 3 (Data Science 3.0)",
   "language": "python",
   "name": "python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-west-2:236514542706:image/sagemaker-data-science-310-v1"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
