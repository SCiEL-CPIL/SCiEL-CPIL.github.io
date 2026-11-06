def _greedy(order):
    """
    Run greedy construction with a given order of candidates.
    Returns the resulting set.
    """
    current = set()
    for k in order:
        if _time_remaining() < 10.0:
            break
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
    return current

def _local_search(S):
    """
    Improve set S using local search (swaps).
    """
    current = set(S)
    improved = True
    while improved and _time_remaining() > 15.0:
        improved = False
        # 1. Try to add any missing element
        for k in range(1, 25):
            if k not in current:
                new_set = current | {k}
                if _is_subsum_free(new_set):
                    current = new_set
                    improved = True
                    break # Restart loop to check for more additions
        
        if _time_remaining() < 15.0:
            break

        # 2. Try to remove x and add y, z (increase cardinality by 1)
        elements = list(current)
        removed_any = False
        for i in range(len(elements)):
            if _time_remaining() < 15.0:
                break
            x = elements[i]
            for y in range(1, 25):
                if y in current:
                    continue
                for z in range(y+1, 25):
                    if z in current:
                        continue
                    new_set = (current - {x}) | {y, z}
                    if _is_subsum_free(new_set):
                        current = new_set
                        improved = True
                        removed_any = True
                        break
                if removed_any:
                    break
            if removed_any:
                break
        
        if not removed_any and not improved:
             break

    return current

def construct():
    """
    Return a nonempty subset of [1,24] that is reciprocal-subsum-free,
    maximizing cardinality.
    """
    best_set = set()
    
    # Try different greedy orders
    candidates = [
        list(range(24, 0, -1)), # Descending
        list(range(1, 25)),     # Ascending
    ]
    
    # Add random permutations
    random.seed(42)
    for _ in range(20):
        perm = list(range(1, 25))
        random.shuffle(perm)
        candidates.append(perm)
    
    for order in candidates:
        if _time_remaining() < 20.0:
            break
        S = _greedy(order)
        if len(S) > len(best_set):
            best_set = S
            
    # Local search on the best set
    if _time_remaining() > 15.0:
        best_set = _local_search(best_set)
        
    if not best_set:
        return [24]
        
    return list(best_set)
