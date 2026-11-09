def construct():
    p = 1009
    # Precompute modular inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    best_len = 0
    best_start = 1
    
    # Iterate over all possible start positions
    for s in range(1, p):
        # Find the minimum inverse value I(x) for x in [s, p-1] such that I(x) >= s
        # This determines the first "conflict" that would invalidate the interval
        min_inv = p  # Initialize with a value larger than any possible inverse
        for x in range(s, p):
            if inv[x] >= s:
                if inv[x] < min_inv:
                    min_inv = inv[x]
        
        # The maximum valid end is min_inv - 1
        e = min_inv - 1
        
        # Check if a valid non-empty interval exists
        if e >= s:
            current_len = e - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
                
    return [best_start, best_len]
