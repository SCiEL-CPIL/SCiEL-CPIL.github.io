def _local_search_best_swap(individual, max_iters=10):
    """
    Intensive local search: try all possible single swaps and pick the best.
    """
    best_ind = list(individual)
    best_mask = 0
    for x in best_ind:
        best_mask |= (1 << (x - 1))
    best_fit = _fitness_from_mask(best_mask)
    ind_set = set(best_ind)
    
    for _ in range(max_iters):
        improved = False
        best_new_fit = best_fit
        best_swap = None
        
        for i in range(32):
            old_val = best_ind[i]
            old_bit = 1 << (old_val - 1)
            
            for new_val in range(1, 513):
                if new_val in ind_set:
                    continue
                
                new_bit = 1 << (new_val - 1)
                temp_mask = (best_mask & ~old_bit) | new_bit
                new_fit = _fitness_from_mask(temp_mask)
                
                if new_fit > best_new_fit:
                    best_new_fit = new_fit
                    best_swap = (i, old_val, new_val, temp_mask)
        
        if best_swap is not None:
            improved = True
            i, old_val, new_val, new_mask = best_swap
            best_mask = new_mask
            ind_set.remove(old_val)
            ind_set.add(new_val)
            best_ind[i] = new_val
            best_fit = best_new_fit
        else:
            break
            
    return best_ind, best_fit
