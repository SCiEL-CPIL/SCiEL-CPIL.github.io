def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145 # Safety margin for 150s limit
    
    best_rows = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Multiple random restarts with local search
    while time.time() < deadline - 10:
        rows = _greedy_construct(n_rows, k, rng)
        if rows is None:
            continue
            
        if not _validate(rows):
            continue
            
        opt_rows = _local_search_reduce_max(rows, deadline - 5)
        
        score = _max_entry(opt_rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in opt_rows]
            
        if best_score <= 105:
            break

    if best_rows is None:
        rows = _greedy_construct(n_rows, k, random.Random(0))
        if rows:
            best_rows = rows
        else:
            best_rows = [[0, 1, 3, 7, 15, 31]] * n_rows

    return best_rows
