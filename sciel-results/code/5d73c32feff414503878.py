<<<<<<< SEARCH
def _local_search_swap(subset, sums, universe, deadline):
    """Try swap moves to improve the set."""
    best_subset = subset[:]
    best_sums = sums
    best_size = len(subset)
    
    subset_set = set(subset)
    
    # Try all possible single swaps
    for i, out_elem in enumerate(subset):
        if time.time() > deadline:
            break
        for in_elem in universe:
            if in_elem in subset_set:
                continue
            
            # Build new set
            new_subset = subset[:i] + subset[i+1:] + [in_elem]
            
            if _is_dissociated_fast(new_subset):
                new_sums = _build_sums_set(new_subset)
                if len(new_subset) > best_size:
                    best_subset = new_subset
                    best_sums = new_sums
                    best_size = len(new_subset)
                    subset_set = set(best_subset)
    
    return best_subset, best_sums, best_size
=======
def _local_search_swap(subset, sums, universe, deadline):
    """Try swap moves: remove one element, try to add a different one."""
    best_subset = subset[:]
    best_size = len(subset)
    subset_set = set(subset)
    universe_set = set(universe)
    
    for i, out_elem in enumerate(subset):
        if time.time() > deadline:
            break
        # Build subset without out_elem
        base = subset[:i] + subset[i+1:]
        base_set = set(base)
        base_sums = _build_sums_set(base)
        base_mod1, base_mod2 = _build_dual_mod_sums(base)
        
        for in_elem in universe:
            if in_elem in base_set:
                continue
            v = _LCM // in_elem
            v_mod1 = _MOD_WEIGHTS1[in_elem]
            v_mod2 = _MOD_WEIGHTS2[in_elem]
            
            if not _check_add_mod_dual(base_mod1, base_mod2, v_mod1, v_mod2):
                continue
            if not _check_add_fast(base_sums, v):
                continue
            
            new_subset = base + [in_elem]
            if len(new_subset) > best_size:
                best_subset = new_subset
                best_size = len(new_subset)
    
    best_sums = _build_sums_set(best_subset)
    return best_subset, best_sums, best_size
>>>>>>> REPLACE

