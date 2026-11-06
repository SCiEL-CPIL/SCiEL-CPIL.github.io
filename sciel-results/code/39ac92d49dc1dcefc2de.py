import inspect
try:
    import autoresearch_suite.science_tcs_expansion as m
    print("MODULE:", m.__file__)
    for name in dir(m):
        if not name.startswith('__'):
            print("  ", name)
except Exception as e:
    print("import err", e)
