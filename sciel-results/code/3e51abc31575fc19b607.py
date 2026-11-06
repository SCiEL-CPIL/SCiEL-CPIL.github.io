def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s limit
    
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
        
    # Calculate L_x = min(x, inv[x]) and H_x = max(x, inv[x])
    lo = [0] * (n + 1)
    hi = [0] * (n + 1)
    for x in range(1, n + 1):
        ix = inv[x]
        lo[x] = min(x, ix)
        hi[x] = max(x, ix)
        
    # We need M(s) = min { H_x | L_x >= s }
    # Since L_x is in [1, n], we can use an array of size n+2
    # min_hi_for_lo[L] stores the minimum H_x for all x with L_x == L
    min_hi_for_lo = [float('inf')] * (n + 2)
    
    for x in range(1, n + 1):
        L = lo[x]
        H = hi[x]
        if H < min_hi_for_lo[L]:
            min_hi_for_lo[L] = H
            
    # Suffix minimum array
    # suff_min[s] = min(min_hi_for_lo[s], min_hi_for_lo[s+1], ..., min_hi_for_lo[n])
    suff_min = [float('inf')] * (n + 2)
    current_min = float('inf')
    for L in range(n, 0, -1):
        if min_hi_for_lo[L] < current_min:
            current_min = min_hi_for_lo[L]
        suff_min[L] = current_min
        
    best_start = 1
    best_length = 0
    
    # Find the start s that maximizes the valid interval length
    for s in range(1, n + 1):
        if time.time() > deadline:
            break
            
        # M(s) is the smallest H_x for any pair where both elements are >= s
        # The interval [s, e] is valid if e < M(s)
        # So max e = M(s) - 1
        max_e = suff_min[s] - 1
        
        # Cap at n
        if max_e > n:
            max_e = n
            
        if max_e >= s:
            length = max_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
                
    # Optional: Verification step (fast)
    # s = best_start
    # e = s + best_length - 1
    # for x in range(s, e + 1):
    #     if s <= inv[x] <= e:
    #         return [2, 1] # Fallback
        
    return [best_start, best_length]
