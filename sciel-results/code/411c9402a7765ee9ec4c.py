# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = 1008
    # Precompute modular inverses in [1, 1008]
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_len = 0
    best_start = 1
    
    # For each start s, find the max valid end e
    for s in range(1, n + 1):
        # Early termination: if max possible length from s is not better than best, stop
        if n - s + 1 <= best_len:
            break
            
        # Self-inverse check for start (1 and 1008)
        if s == 1 or s == 1008:
            continue
            
        e = s
        # We extend e as long as valid
        # Condition to break:
        # 1. e_candidate is self-inverse (only 1008 in this range > 1)
        # 2. inv[e_candidate] is in [s, e_candidate - 1]
        
        for e_candidate in range(s + 1, n + 1):
            if e_candidate == 1008:
                break
            inv_ec = inv[e_candidate]
            if inv_ec >= s and inv_ec < e_candidate:
                break
            e = e_candidate
        
        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
