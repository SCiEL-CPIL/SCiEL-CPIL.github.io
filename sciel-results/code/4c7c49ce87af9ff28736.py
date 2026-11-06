def _inspect():
    import inspect
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    print(inspect.getsource(pebble_baseline))
    try:
        import autoresearch_suite.science_tcs_expansion as m
        print([n for n in dir(m) if not n.startswith('__')])
    except Exception as e:
        print("ERR", e)
