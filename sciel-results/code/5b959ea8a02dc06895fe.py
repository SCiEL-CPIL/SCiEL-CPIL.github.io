def construct():
    """Find the longest interval [start, start+length-1] within [1, 1008] such that
    for every x, y in the interval (including x=y), x*y % 1009 != 1."""
    p = 1009
    
    # Precompute modular inverses: inv[i] = i^(-1) mod p
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    best_len = 0
    best_start = 2  # Default safe start
    
    # Iterate over all possible starts
    for s in range(1, p):
        # Pruning: if max possible length from this start is <= best, stop
        if p - 1 - s + 1 <= best_len:
            break
        
        for e in range(s, p):
            # Check diagonal constraint: e^2 mod p != 1
            # This happens when e == 1 or e == p-1 (1008)
            if e == 1 or e == p - 1:
                break
            
            # Check off-diagonal constraint:
            # If inv[e] is in [s, e-1], then both e and inv[e] are in the interval
            # and their product is 1 mod p, which is forbidden.
            # Note: inv[e] cannot be e (since e^2 != 1 mod p for e not 1, p-1)
            if s <= inv[e] < e:
                break
            
            current_len = e - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
    
    return [best_start, best_len]
