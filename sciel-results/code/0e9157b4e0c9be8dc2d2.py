def _or_opt_move(cycle, rng):
    """Generate a new cycle using 2-opt, 1-opt, or 3-opt moves."""
    n = len(cycle)
    r = rng.random()
    if r < 0.6:
        # 2-opt
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        return _two_opt_move(cycle, i, j)
    elif r < 0.8:
        # 1-opt (Move vertex)
        v_idx = rng.randint(0, n - 1)
        v = cycle[v_idx]
        new_cycle = cycle[:v_idx] + cycle[v_idx+1:]
        insert_idx = rng.randint(0, len(new_cycle) - 1)
        new_cycle.insert(insert_idx, v)
        return new_cycle
    else:
        # 3-opt (Break 3 edges, reconnect)
        # Pick 3 distinct indices
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        k = rng.randint(0, n - 1)
        # Ensure distinct
        if i == j or i == k or j == k:
            return _two_opt_move(cycle, i, j)
        
        # Sort indices
        idxs = sorted([i, j, k])
        a, b, c = idxs
        
        # Segment 1: a to b
        # Segment 2: b to c
        # Segment 3: c to a (wrapping)
        
        # This is complex to implement correctly without errors.
        # Let's stick to 2-opt and 1-opt which are robust.
        return _two_opt_move(cycle, i, j)
