def _greedy_search(order):
    S = []
    sums = {0}
    
    for x in order:
        new_sums = {s + x for s in sums}
        
        # Check for duplicate sums (collision)
        if new_sums & sums:
            continue
            
        candidate_sums = sums.union(new_sums)
        candidate_sums.discard(0)
        
        if _is_valid_sums(candidate_sums):
            S.append(x)
            sums = candidate_sums
            
    return S
