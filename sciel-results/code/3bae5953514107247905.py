"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Precompute modular inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
        
    best_len = 0
    best_start = 1
    
    # 1 and 1008 are self-inverses, so they cannot be in the interval.
    # We only consider start and end points in [2, 1007]
    for s in range(2, p - 1):
        # Optimization: if the maximum possible length from s is <= best_len, we can stop
        # Max length is (p-2) - s + 1 = p - 1 - s
        if (p - 2) - s + 1 <= best_len:
            break
            
        for e in range(s, p - 1):
            # Check if adding e creates a forbidden pair with an existing element in [s, e-1]
            # The pair is (e, inv[e]). We need inv[e] to be in [s, e-1].
            # Note: inv[e] cannot be e because e != 1 and e != p-1.
            if s <= inv[e] < e:
                break
            
            current_len = e - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
                
    return [best_start, best_len]
# EVOLVE-BLOCK-END
