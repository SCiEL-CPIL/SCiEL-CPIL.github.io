<<<<<<< SEARCH
    # Main Logic
    rng = np.random.default_rng(42)
    
    # 1. Greedy Initialization
    initial_code = greedy_init(rng)
    
    # 2. Local Search (SA)
    best_code, best_size = local_search(initial_code, rng, deadline)
    
    # 3. Final Polish
    polished_code, polished_size = final_polish(best_code)
    
    if polished_size < best_size:
        best_code = polished_code
        best_size = polished_size
        
    # If we have time, run a second restart with different seed?
    # Given the tight time, one good restart with strong SA is usually sufficient.
    # Let's check if we have significant time left.
    if time.time() < deadline - 10:
        rng2 = np.random.default_rng(123)
        initial_code2 = greedy_init(rng2)
        code2, size2 = local_search(initial_code2, rng2, deadline)
        polished_code2, polished_size2 = final_polish(code2)
        
        if polished_size2 < best_size:
            best_code = polished_code2
            best_size = polished_size2
=======
    # Main Logic
    rng = np.random.default_rng(42)
    
    # 1. Greedy Initialization
    initial_code = greedy_init(rng)
    
    # 2. Local Search (SA)
    best_code, best_size = local_search(initial_code, rng, deadline)
    
    # 3. Final Polish
    polished_code, polished_size = final_polish(best_code)
    
    if polished_size < best_size:
        best_code = polished_code
        best_size = polished_size
        
    # If we have time, run additional restarts
    if time.time() < deadline - 15:
        rng2 = np.random.default_rng(123)
        initial_code2 = greedy_init(rng2)
        code2, size2 = local_search(initial_code2, rng2, deadline)
        polished_code2, polished_size2 = final_polish(code2)
        
        if polished_size2 < best_size:
            best_code = polished_code2
            best_size = polished_size2
    
    if time.time() < deadline - 15:
        rng3 = np.random.default_rng(777)
        initial_code3 = greedy_init(rng3)
        code3, size3 = local_search(initial_code3, rng3, deadline)
        polished_code3, polished_size3 = final_polish(code3)
        
        if polished_size3 < best_size:
            best_code = polished_code3
            best_size = polished_size3
>>>>>>> REPLACE
