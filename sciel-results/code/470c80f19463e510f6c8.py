<<<<<<< SEARCH
    # Run multiple restarts with different seeds
    best_overall = None
    best_overall_size = float('inf')
    
    # Strategy: Try several restarts, each with greedy + local search
    # Allocate time: ~15s per restart for greedy, rest for local search
    num_restarts = 4
    seeds = [42, 123, 456, 789]
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        seed = seeds[restart % len(seeds)]
        
        # Generate greedy code
        code, cov_count, is_code = generate_greedy_code(seed)
        
        if not code:
            continue
        
        # Time remaining for local search
        time_left = deadline - time.time() - 5.0  # 5s safety margin
        if time_left < 5.0:
            break
        
        # Perform local search
        code, cov_count, is_code = perform_local_search(code, cov_count, is_code, time_left, seed=seed)
        
        # Verify and update best
        if len(code) < best_overall_size:
            # Verify coverage
            final_cov = np.zeros(total_words, dtype=np.int32)
            for c in code:
                final_cov[neighbors[c]] += 1
            
            if np.all(final_cov > 0):
                best_overall = code
                best_overall_size = len(code)
    
    # If no valid solution found, use last greedy
    if best_overall is None:
        code, _, _ = generate_greedy_code(42)
        best_overall = code
    
    return [list(digits_of[i]) for i in best_overall]
=======
    # Run multiple restarts with different seeds
    best_overall = None
    best_overall_size = float('inf')
    
    # Strategy: Try several restarts, each with greedy + local search
    # Allocate time more aggressively to local search
    num_restarts = 5
    seeds = [42, 123, 456, 789, 1011]
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        seed = seeds[restart % len(seeds)]
        
        # Generate greedy code
        code, cov_count, is_code = generate_greedy_code(seed)
        
        if not code:
            continue
        
        # Time remaining for local search
        time_left = deadline - time.time() - 5.0  # 5s safety margin
        if time_left < 3.0:
            break
        
        # Perform local search
        code, cov_count, is_code = perform_local_search(code, cov_count, is_code, time_left, seed=seed)
        
        # Verify and update best
        if len(code) < best_overall_size:
            # Verify coverage
            final_cov = np.zeros(total_words, dtype=np.int32)
            for c in code:
                final_cov[neighbors[c]] += 1
            
            if np.all(final_cov > 0):
                best_overall = code
                best_overall_size = len(code)
    
    # If no valid solution found, use last greedy
    if best_overall is None:
        code, _, _ = generate_greedy_code(42)
        best_overall = code
    
    return [list(digits_of[i]) for i in best_overall]
>>>>>>> REPLACE
