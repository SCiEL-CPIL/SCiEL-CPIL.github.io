<<<<<<< SEARCH
    # Multiple random restarts to escape local optima
    best_solution = None
    best_size = float('inf')
    
    num_restarts = 5
    restart_budget = (deadline - start_time) / (num_restarts + 1)
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Create a perturbed greedy solution
        if r == 0:
            # First restart: pure greedy
            current_set = greedy_construct()
        else:
            # Subsequent restarts: start from best found, then perturb
            if best_solution is not None:
                current_set = set(best_solution)
                # Remove 2-4 random blocks and re-greedy to fill gaps
                n_remove = min(2 + r, len(current_set) - 1)
                to_remove = random.sample(list(current_set), n_remove)
                for b in to_remove:
                    current_set.discard(b)
                
                # Re-greedy to restore coverage
                uncovered = (1 << num_triples) - 1
                for b in current_set:
                    uncovered &= ~block_masks[b]
                
                while uncovered:
                    if time.time() > deadline - 10:
                        break
                    best_idx = -1
                    best_gain = 0
                    for i in range(num_blocks):
                        if i in current_set:
                            continue
                        gain = (block_masks[i] & uncovered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = i
                    if best_idx == -1:
                        break
                    current_set.add(best_idx)
                    uncovered &= ~block_masks[best_idx]
            else:
                current_set = greedy_construct()
        
        # Local search
        current_set = local_search(current_set)
        
        # Update best
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_solution = set(current_set)
    
    # Final local search on the best solution
    if best_solution is not None:
        best_solution = local_search(best_solution)
=======
    # Multiple random restarts to escape local optima
    best_solution = None
    best_size = float('inf')
    
    num_restarts = 8
    restart_budget = (deadline - start_time - 30) / (num_restarts + 1)  # Reserve 30s for final polish
    
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Create a perturbed greedy solution
        if r == 0:
            # First restart: pure greedy
            current_set = greedy_construct()
        else:
            # Subsequent restarts: start from best found, then perturb
            if best_solution is not None:
                current_set = set(best_solution)
                # Remove 2-5 random blocks and re-greedy to fill gaps
                n_remove = min(2 + r, len(current_set) - 1)
                to_remove = random.sample(list(current_set), n_remove)
                for b in to_remove:
                    current_set.discard(b)
                
                # Re-greedy to restore coverage
                uncovered = (1 << num_triples) - 1
                for b in current_set:
                    uncovered &= ~block_masks[b]
                
                while uncovered:
                    if time.time() > deadline - 30:
                        break
                    best_idx = -1
                    best_gain = 0
                    for i in range(num_blocks):
                        if i in current_set:
                            continue
                        gain = (block_masks[i] & uncovered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = i
                    if best_idx == -1:
                        break
                    current_set.add(best_idx)
                    uncovered &= ~block_masks[best_idx]
            else:
                current_set = greedy_construct()
        
        # Local search
        current_set = local_search(current_set)
        
        # Update best
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_solution = set(current_set)
    
    # Final local search on the best solution with more rounds
    if best_solution is not None:
        best_solution = local_search(best_solution, max_rounds=100)
        # One more pass specifically for 2-swaps
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            if not try_2swap(best_solution):
                break
            try_remove_redundant(best_solution)
>>>>>>> REPLACE
