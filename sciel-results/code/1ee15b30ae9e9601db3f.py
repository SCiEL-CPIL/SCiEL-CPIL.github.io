import os, sys, glob, inspect
print("CWD:", os.getcwd())
print("PATH:", sys.path)
# find the module
for base in sys.path:
    p = os.path.join(base, "autoresearch_suite")
    if os.path.isdir(p):
        print("FOUND suite dir:", p)
        for f in sorted(os.listdir(p)):
            print("   ", f)
