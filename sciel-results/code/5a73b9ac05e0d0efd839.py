<<<<<<< SEARCH
def _exhaustive_small_search(max_val=20, max_size=5):
    """Find the best small sets by exhaustive search."""
    best_set = [0, 1, 2, 5]
    best_score = _compute_score(best_set)
    
    range_val = range(max_val + 1)
    
    for size in range(3, max_size + 1):
        for comb in combinations(range_val, size):
            if 0 not in comb:
                continue
            s = list(comb)
            score = _compute_score(s)
            if score > best_score:
                best_score = score
                best_set = s
                
    return best_set, best_score
=======
def _exhaustive_small_search(max_val=20, max_size=5):
    """Find the best small sets by exhaustive search."""
    best_set = [0, 1, 2, 5]
    best_score = _compute_score(best_set)
    
    range_val = range(max_val + 1)
    
    for size in range(3, max_size + 1):
        for comb in combinations(range_val, size):
            if 0 not in comb:
                continue
            s = list(comb)
            score = _compute_score(s)
            if score > best_score:
                best_score = score
                best_set = s
                
    return best_set, best_score
>>>>>>> REPLACE
