    best_set = None
    best_size = 0
    
    seed_val = 0
    while time.time() < deadline - 2: 
        current_set = greedy_run(seed_val, best_size)
        size = len(current_set)
        if best_set is None or size > best_size:
            best_size = size
            best_set = current_set
        seed_val += 1
        
    if best_set is None:
        # Fallback: run once synchronously?
        # The loop should have run at least once if time permits.
        # But if deadline is very close, it might not.
        # Let's ensure at least one run.
        current_set = greedy_run(0, 0)
        best_set = current_set
        best_size = len(current_set)

    return list(best_set)
