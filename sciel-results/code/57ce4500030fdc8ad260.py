<<<<<<< SEARCH
    # Strategy 3+: Random restarts with weighted greedy + Local Search
    time_elapsed = time.time() - start_time
    time_remaining = deadline - time_elapsed
    
    # Run LS for a few restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Randomized greedy
        sel = greedy_cover(use_weighted=True, random_tiebreak=True, rng=rng, top_k=5)
        if sel:
            peeled = peel_redundant(sel)
            size = len(peeled)
            
            # Apply Local Search
            ls_time_limit = deadline - 5.0
            ls_result = local_search(peeled, rng, ls_time_limit)
            ls_size = len(ls_result)
            
            if ls_size < size:
                size = ls_size
                peeled = ls_result
            
            if size < best_size:
                best_size = size
                best_solution = peeled
=======
    # Strategy 3+: Random restarts with weighted greedy + Local Search
    time_elapsed = time.time() - start_time
    time_remaining = deadline - time_elapsed
    
    # Run LS for a few restarts
    num_restarts = 12
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Randomized greedy with varying top_k
        top_k_val = rng.choice([3, 5, 8, 12])
        sel = greedy_cover(use_weighted=True, random_tiebreak=True, rng=rng, top_k=top_k_val)
        if sel:
            peeled = peel_redundant(sel)
            size = len(peeled)
            
            # Apply Local Search
            ls_time_limit = deadline - 5.0
            ls_result = local_search(peeled, rng, ls_time_limit)
            ls_size = len(ls_result)
            
            if ls_size < size:
                size = ls_size
                peeled = ls_result
            
            if size < best_size:
                best_size = size
                best_solution = peeled
>>>>>>> REPLACE
