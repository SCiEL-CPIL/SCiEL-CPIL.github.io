<<<<<<< SEARCH
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Run greedy instances and pick the best after pruning
    best_code = None
    best_len = float('inf')
    
    # Run multiple greedy instances with different seeds
    num_greedy = 8
    for seed in range(num_greedy):
        if time.time() > deadline - 40:
            break
        selected = greedy_set_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
            
    if best_code is None:
        best_code = greedy_set_cover(0)
        best_code = prune(best_code)
        best_len = len(best_code)
    
    # Local Search on the best code
    best_code = local_search_optimize(best_code, deadline)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
=======
    start_time = time.time()
    deadline = start_time + 115.0

    # Run greedy instances and pick the best after pruning
    best_code = None
    best_len = float('inf')

    # Run multiple greedy instances with different seeds
    num_greedy = 10
    for seed in range(num_greedy):
        if time.time() > deadline - 50:
            break
        selected = greedy_set_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected

    if best_code is None:
        best_code = greedy_set_cover(0)
        best_code = prune(best_code)
        best_len = len(best_code)

    # Local Search on the best code
    best_code = local_search_optimize(best_code, deadline)

    # Final polish: one more round of local search with a different RNG seed
    if time.time() < deadline - 2:
        best_code = local_search_optimize(best_code, deadline)

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result
>>>>>>> REPLACE
