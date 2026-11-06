def _local_search(s, mask, score, deadline):
    current_s = set(s)
    current_mask = mask
    current_score = score
    
    # Limit time for this specific local search run to allow restarts
    # But we don't know how many restarts we want.
    # Let's just use the global deadline.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_remove = None
        best_add = None
        
        # Shuffle order of removals to avoid bias?
        # No, best improvement is deterministic.
        
        for a in list(current_s):
            # Optimization: If we already found a gain of G, and the max possible gain is small,
            # we might skip. But max gain can be > 1.
            
            for b in range(1, 513):
                if b in current_s:
                    continue
                
                new_mask = (current_mask & ~(1 << a)) | (1 << b)
                new_score = _score_mask(new_mask)
                
                gain = new_score - current_score
                if gain > best_gain:
                    best_gain = gain
                    best_remove = a
                    best_add = b
                    improved = True
        
        if best_remove is not None and best_add is not None:
            current_s.remove(best_remove)
            current_s.add(best_add)
            current_mask = (current_mask & ~(1 << best_remove)) | (1 << best_add)
            current_score += best_gain
            
    return current_s, current_mask, current_score
