import os, sys
print("cwd", os.getcwd())
try:
    import autoresearch_suite.science_tcs_expansion as m
    print("module", m.__file__)
    print("names", [n for n in dir(m) if not n.startswith('__')])
except Exception as e:
    print("ERR", repr(e))
