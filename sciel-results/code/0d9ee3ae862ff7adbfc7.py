def construct():
    import time
    deadline = time.time() + 85  # safety margin before 90s kill
    
    n = 1008
    p = 1009
    
    # Precompute modular inverses mod 1009 for x in [1, 1008]
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
        
    # lo[x] = min(x, inv[x]), hi[x] = max(x, inv[x])
    # A pair (x, inv[x]) blocks an interval [s, e] if:
    # 1. x is in [s, e] AND inv[x] is in [s, e]
    # This is equivalent to: s <= min(x, inv[x]) AND e >= max(x, inv[x])
    # So for a fixed s, the constraint is e < min{ max(x, inv[x]) | min(x, inv[x]) >= s }
    
    # We want to find for each s, the minimum hi[x] among all x such that lo[x] >= s.
    # Let M[s] = min{ hi[x] | lo[x] >= s }.
    # Then max valid e for start s is M[s] - 1.
    # If M[s] is infinity (no such x), then max valid e is n.
    
    # Initialize M with infinity (represented by n+1)
    M = [n + 1] * (n + 2)
    
    # Group by lo value? Or just iterate?
    # Since n is small (1008), we can just do a simple loop for each s?
    # O(n^2) is 10^6 operations, which is very fast in Python (<0.1s).
    # However, we can do it in O(n log n) or O(n) with suffix min.
    
    # Let's stick to the O(n) suffix min approach but cleaner.
    # min_hi_for_lo[v] = min{ hi[x] | lo[x] == v }
    min_hi_for_lo = [n + 1] * (n + 2)
    
    for x in range(1, n + 1):
        ix = inv[x]
        lo_val = min(x, ix)
        hi_val = max(x, ix)
        if hi_val < min_hi_for_lo[lo_val]:
            min_hi_for_lo[lo_val] = hi_val
            
    # Compute suffix minimums
    # M[s] = min( min_hi_for_lo[s], M[s+1] )
    # Base case: M[n+1] = n+1
    M = [n + 1] * (n + 2)
    running_min = n + 1
    for s in range(n, 0, -1):
        if min_hi_for_lo[s] < running_min:
            running_min = min_hi_for_lo[s]
        M[s] = running_min
        
    best_start = 1
    best_length = 0
    
    for s in range(1, n + 1):
        # M[s] is the smallest hi[x] such that lo[x] >= s.
        # Any interval starting at s must end at e < M[s].
        # So max_e = M[s] - 1.
        # If M[s] == n+1, it means there are no pairs with lo[x] >= s, so we can go up to n.
        max_e = M[s] - 1
        
        if max_e >= s:
            length = max_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
                
        if time.time() > deadline:
            break
            
    return [best_start, best_length]
