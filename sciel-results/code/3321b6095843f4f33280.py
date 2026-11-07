<<<<<<< SEARCH
def _try_remove_add(s_set, start_time, deadline_margin=5.0):
    """Try removing one element and adding one or more others."""
    s_set = set(s_set)
    best_s = set(s_set)
    best_len = len(s_set)
    
    for remove_el in sorted(s_set):
        if time.time() - start_time > MAX_TIME - deadline_margin:
            break
        remaining_set = s_set - {remove_el}
        candidates = [x for x in range(1, N+1) if x not in remaining_set]
        candidates.sort()
        
        # Try to add as many as possible from remaining_set
        current = set(remaining_set)
        for cand in candidates:
            if time.time() - start_time > MAX_TIME - deadline_margin:
                break
            new_s = current | {cand}
            if _fast_validity_check(new_s):
                current = new_s
        
        if len(current) > best_len:
            best_s = current
            best_len = len(current)
    
    return sorted(best_s)
=======
def _try_remove_add(s_set, start_time, deadline_margin=5.0):
    """Try removing one element and adding one or more others."""
    s_set = set(s_set)
    best_s = set(s_set)
    best_len = len(s_set)
    
    for remove_el in sorted(s_set):
        if time.time() - start_time > MAX_TIME - deadline_margin:
            break
        remaining_set = s_set - {remove_el}
        candidates = [x for x in range(1, N+1) if x not in remaining_set]
        candidates.sort()
        
        # Try to add as many as possible from remaining_set
        current = set(remaining_set)
        for cand in candidates:
            if time.time() - start_time > MAX_TIME - deadline_margin:
                break
            new_s = current | {cand}
            if _fast_validity_check(new_s):
                current = new_s
        
        if len(current) > best_len:
            best_s = current
            best_len = len(current)
    
    return sorted(best_s)
>>>>>>> REPLACE

