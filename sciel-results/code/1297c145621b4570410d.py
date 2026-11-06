def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 2
    best_length = 0
    
    # For each start s, compute max valid end
    for s in range(1, n + 1):
        # Find max e such that no x in [s,e] has inv[x] in [s,e]
        # e must be < max(x, inv[x]) for all x >= s with inv[x] >= s
        max_e = n  # 1008
        
        for x in range(s, n + 1):
            ix = inv[x]
            if ix >= s:
                # Conflict when e >= max(x, ix)
                conflict = max(x, ix)
                if conflict - 1 < max_e:
                    max_e = conflict - 1
        
        if max_e >= s:
            length = max_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
    
    return [best_start, best_length]
