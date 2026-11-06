import os, sys, json
print("CWD:", os.getcwd())
print("PATH entries:")
for p in sys.path:
    print(" ", p)
# Try to find the suite
try:
    import autoresearch_suite
    print("suite file:", autoresearch_suite.__file__)
    print("suite dir:", os.path.dirname(autoresearch_suite.__file__))
except Exception as e:
    print("no suite:", e)
