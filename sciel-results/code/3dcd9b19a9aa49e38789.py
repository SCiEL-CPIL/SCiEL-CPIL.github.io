def construct():
    import time
    start_time = time.time()
    
    MOD = 1009
    N = 1008
    
    inv = [0] * MOD
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_len = 0
    best_start = 2
    
    in_window = [False] * MOD
    R = 1
    
    for L in range(2, 1008):
        if time.time() - start_time > 80:
            break
        
        if R < L - 1:
            R = L - 1
        
        # Extend R as far as possible while maintaining validity
        while R + 1 <= 1007:
            candidate = R + 1
            p = inv[candidate]
            
            # Can we add candidate without creating a conflict?
            # Conflict if p is in window (including p == candidate)
            if not in_window[p]:
                in_window[candidate] = True
                R = candidate
            else:
                break
        
        # Record best
        if R >= L:
            length = R - L + 1
            if length > best_len:
                best_len = length
                best_start = L
        
        # Remove L from window
        if L <= R:
            in_window[L] = False
    
    return [best_start, best_len]
