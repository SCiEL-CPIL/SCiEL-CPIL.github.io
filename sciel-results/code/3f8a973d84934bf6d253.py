<<<<<<< SEARCH
    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct_vectorized(seed)
        code = reduce_code_fast(code)
        
        if len(code) < best_size or best_code is None:
            if time.time() < deadline - 10:
                code = local_search_improve(code, deadline - 5)
                code = reduce_code_fast(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        
        seed += 1
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
=======
    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    ls_time_budget = 30  # seconds allocated for local search per promising solution
    
    while time.time() < deadline - 5:
        # Adaptive: spend more time on local search as we approach deadline
        remaining = deadline - time.time()
        if remaining < ls_time_budget:
            ls_budget = max(5, int(remaining * 0.5))
        else:
            ls_budget = ls_time_budget
        
        code = greedy_construct_vectorized(seed)
        code = reduce_code_fast(code)
        
        # Only do local search if this looks promising or we have time
        if len(code) <= best_size + 5 and time.time() < deadline - ls_budget:
            ls_deadline = time.time() + ls_budget
            if ls_deadline > deadline - 3:
                ls_deadline = deadline - 3
            code = local_search_improve(code, ls_deadline)
            code = reduce_code_fast(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            # If we found a good solution, do extra local search
            if time.time() < deadline - 15:
                extra_deadline = time.time() + min(15, deadline - time.time() - 3)
                code = local_search_improve(code, extra_deadline)
                code = reduce_code_fast(code)
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code
        
        seed += 1
    
    # Final polish: if we have time, try to improve the best solution further
    if best_code is not None and time.time() < deadline - 3:
        final_deadline = deadline - 1
        polished = local_search_improve(best_code, final_deadline)
        polished = reduce_code_fast(polished)
        if len(polished) < best_size:
            best_code = polished
            best_size = len(best_code)
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
>>>>>>> REPLACE
