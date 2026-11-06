def pebble(dag, fast_memory_budget):
    start = time.time()
    DEADLINE = 85.0
    # Try baseline first (safe fallback)
    ops_base = None
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        ops_base = pebble_baseline(dag, fast_memory_budget)
    except Exception:
        ops_base = None
    # Try my algorithm
    ops_mine = None
    try:
        vertices, parents_dict, inputs, outputs = _parse_dag(dag)
        ops_mine, transfers_mine = _greedy_pebble(vertices, parents_dict, inputs, outputs, fast_memory_budget, start, DEADLINE)
        # verify
        if not _verify(ops_mine, parents_dict, inputs, outputs, fast_memory_budget, vertices):
            ops_mine = None
    except Exception:
        ops_mine = None
    # Decide
    if ops_mine is not None and ops_base is not None:
        t_mine = sum(1 for op, v in ops_mine if op in ('load','store'))
        t_base = sum(1 for op, v in ops_base if op in ('load','store'))
        if t_mine <= t_base:
            return ops_mine
        else:
            return ops_base
    elif ops_mine is not None:
        return ops_mine
    elif ops_base is not None:
        return ops_base
    else:
        return []
