import sys, importlib
try:
    m = importlib.import_module("autoresearch_suite.science_tcs_expansion")
    print("MODULE OK", [x for x in dir(m) if not x.startswith('__')])
except Exception as e:
    print("IMPORT FAIL", repr(e))
