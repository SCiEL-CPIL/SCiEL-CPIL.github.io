else:
    # Tight budget: process in topological order, store when tight
    for v in topo:
        if v not in needed:
            continue
        if v in inputs:
            load(v)
        else:
            for p in get_parents(v):
                if not is_red(p):
                    if is_blue(p):
                        load(p)
                    else:
                        # Should not happen if we store (not delete)
                        raise ValueError(f"Parent {p} not available for {v}")
            compute(v)
        # If over budget, store the value with smallest last_use (not an output)
        while len(fast_mem) > fast_memory_budget:
            candidates = [x for x in fast_mem if x not in outputs]
            if not candidates:
                break
            x = min(candidates, key=lambda y: last_use.get(y, -1))
            store(x)
    # Store outputs
    for v in outputs:
        if is_red(v):
            store(v)
