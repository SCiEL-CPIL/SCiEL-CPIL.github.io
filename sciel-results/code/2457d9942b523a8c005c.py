<<<<<<< SEARCH
    # Main Loop: Multiple Restarts
    n_restarts = 0
    max_restarts = 20
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        
        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
            
        in_set = local_search(in_set)
        
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 3. Optional: Try a random perturbation of the best set found so far
        if n_restarts > 1 and time.time() < deadline - 5:
            perturbed = best_in_set.copy()
            # Remove 2 random elements
            indices = np.where(perturbed)[0]
            if len(indices) > 2:
                to_remove = rng.choice(indices, size=2, replace=False)
                perturbed[to_remove] = False
            perturbed = local_search(perturbed)
            current_size = perturbed.sum()
            if current_size > best_size:
                best_size = current_size
                best_in_set = perturbed.copy()
=======
    # Main Loop: Multiple Restarts
    n_restarts = 0
    max_restarts = 30
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        
        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 3:
            break
            
        in_set = local_search(in_set)
        
        current_size = int(in_set.sum())
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 3. Optional: Try a random perturbation of the best set found so far
        if n_restarts > 1 and time.time() < deadline - 5:
            perturbed = best_in_set.copy()
            # Remove 3-5 random elements for stronger perturbation
            indices = np.where(perturbed)[0]
            if len(indices) > 5:
                n_rem = rng.integers(3, min(6, len(indices) - 2))
                to_remove = rng.choice(indices, size=n_rem, replace=False)
                perturbed[to_remove] = False
            perturbed = local_search(perturbed)
            current_size = int(perturbed.sum())
            if current_size > best_size:
                best_size = current_size
                best_in_set = perturbed.copy()
>>>>>>> REPLACE
