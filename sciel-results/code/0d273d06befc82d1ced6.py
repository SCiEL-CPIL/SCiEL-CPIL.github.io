<<<<<<< SEARCH
    # Run the search
    current_x = x_init
    current_cc = cc_init
    
    # First pass: Pure hill climbing (removals)
    current_x, current_cost = local_search_hill_climb(current_x, current_cc, max_iters=5000)
    if current_cost < best_len:
        best_len = current_cost
        best_solution_idx = np.where(current_x)[0].tolist()
        
    # Second pass: Hill climbing with swaps (perturbation)
    # Re-initialize cover counts for the current best
    current_cc = get_cover_counts(current_x)
    
    # Run multiple times with different random seeds/perturbations
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
        random.seed(restart)
        # Perturb the current best solution slightly?
        # Or just run the swap-based search on the current best.
        new_x, new_cost = local_search_with_swap(current_x, current_cc, max_iters=2000)
        
        if new_cost < best_len:
            best_len = new_cost
            best_solution_idx = np.where(new_x)[0].tolist()
            current_x = new_x
            current_cc = get_cover_counts(current_x)
            
    # Final cleanup: Remove any redundant blocks again
    final_x = np.zeros(num_k_blocks, dtype=bool)
    for j in best_solution_idx:
        final_x[j] = True
    final_cc = get_cover_counts(final_x)
    
    # One more pass of pure removal to ensure minimality
    final_x, final_cost = local_search_hill_climb(final_x, final_cc, max_iters=1000)
    if final_cost < best_len:
        best_len = final_cost
        best_solution_idx = np.where(final_x)[0].tolist()
=======
    # Run the search
    current_x = x_init
    current_cc = cc_init
    
    # First pass: Pure hill climbing (removals)
    current_x, current_cost = local_search_hill_climb(current_x, current_cc, max_iters=8000)
    if current_cost < best_len:
        best_len = current_cost
        best_solution_idx = np.where(current_x)[0].tolist()
        
    # Second pass: Hill climbing with swaps (perturbation)
    # Re-initialize cover counts for the current best
    current_cc = get_cover_counts(current_x)
    
    # Run multiple times with different random seeds/perturbations
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        random.seed(restart)
        new_x, new_cost = local_search_with_swap(current_x, current_cc, max_iters=3000)
        
        if new_cost < best_len:
            best_len = new_cost
            best_solution_idx = np.where(new_x)[0].tolist()
            current_x = new_x
            current_cc = get_cover_counts(current_x)
            
            # After improvement, do another round of pure removals
            current_x, current_cost = local_search_hill_climb(current_x, current_cc, max_iters=3000)
            if current_cost < best_len:
                best_len = current_cost
                best_solution_idx = np.where(current_x)[0].tolist()
                current_cc = get_cover_counts(current_x)
            
    # Final cleanup: Remove any redundant blocks again
    final_x = np.zeros(num_k_blocks, dtype=bool)
    for j in best_solution_idx:
        final_x[j] = True
    final_cc = get_cover_counts(final_x)
    
    # One more pass of pure removal to ensure minimality
    final_x, final_cost = local_search_hill_climb(final_x, final_cc, max_iters=2000)
    if final_cost < best_len:
        best_len = final_cost
        best_solution_idx = np.where(final_x)[0].tolist()
>>>>>>> REPLACE
