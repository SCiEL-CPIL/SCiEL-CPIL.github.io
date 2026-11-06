def construct():
    import time
    start_time = time.time()
    deadline = start_time + 40
    
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 1
    best_length = 0
    
    # For each start s, compute max valid end
    # An interval [s, e] is valid iff for all x in [s,e], inv[x] is not in [s,e]
    # Equivalently, for all x in [s,e], max(x, inv[x]) > e OR min(x, inv[x]) < s
    # For a fixed s, the constraint is: e < min over x in [s,e] of max(x, inv[x])
    # But e also must be >= s, and x must be in [s,e]
    
    # Better approach: for each s, find the smallest "blocking" position
    # A position x blocks the interval starting at s if inv[x] >= s
    # In that case, e must be < max(x, inv[x]) (since both x and inv[x] would be in [s,e])
    # Wait, that's not quite right. Let me think again.
    
    # For interval [s, e], it's invalid if there exist x, y in [s,e] with x*y % p == 1
    # This means y = inv[x] (mod p), and since x,y in [1,1008], y = inv[x] in [1,1008]
    # So invalid iff there exists x in [s,e] such that inv[x] in [s,e]
    
    # For fixed s, we want max e such that no x in [s,e] has inv[x] in [s,e]
    # For each x, the condition "x in [s,e] AND inv[x] in [s,e]" is equivalent to:
    #   s <= x <= e AND s <= inv[x] <= e
    #   which means s <= min(x, inv[x]) AND max(x, inv[x]) <= e
    # So for fixed s, x blocks the interval if min(x, inv[x]) >= s AND max(x, inv[x]) <= e
    # The interval [s,e] is valid iff for all x: NOT(min(x, inv[x]) >= s AND max(x, inv[x]) <= e)
    # Equivalently: for all x with min(x, inv[x]) >= s, we need max(x, inv[x]) > e
    # So e < min over x with min(x, inv[x]) >= s of max(x, inv[x])
    
    # Precompute for each x: lo[x] = min(x, inv[x]), hi[x] = max(x, inv[x])
    lo = [0] * (n + 1)
    hi = [0] * (n + 1)
    for x in range(1, n + 1):
        ix = inv[x]
        lo[x] = min(x, ix)
        hi[x] = max(x, ix)
    
    # For each s, find min hi[x] over all x with lo[x] >= s
    # Then max valid e = that min - 1
    # We can precompute this efficiently by sorting x by lo[x] and using a suffix minimum of hi[x]
    
    # For each s from 1 to n, we need min hi[x] over x with lo[x] >= s
    # Create array: for each lo value L, what's the min hi among x with lo[x] == L
    # Then suffix min over L
    
    min_hi_for_lo = [float('inf')] * (n + 2)
    for x in range(1, n + 1):
        L = lo[x]
        H = hi[x]
        if H < min_hi_for_lo[L]:
            min_hi_for_lo[L] = H
    
    # Suffix minimum: suff_min[L] = min(min_hi_for_lo[L], min_hi_for_lo[L+1], ..., min_hi_for_lo[n])
    suff_min = [float('inf')] * (n + 2)
    current_min = float('inf')
    for L in range(n, 0, -1):
        if min_hi_for_lo[L] < current_min:
            current_min = min_hi_for_lo[L]
        suff_min[L] = current_min
    
    # Now for each s, max valid e = suff_min[s] - 1
    for s in range(1, n + 1):
        if time.time() > deadline:
            break
        
        max_e = suff_min[s] - 1
        if max_e > n:
            max_e = n
        
        if max_e >= s:
            length = max_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
    
    # Verify the result
    s = best_start
    e = s + best_length - 1
    for x in range(s, e + 1):
        ix = inv[x]
        if s <= ix <= e:
            # This shouldn't happen; fall back to a safe answer
            return [2, 1]
    
    return [best_start, best_length]
