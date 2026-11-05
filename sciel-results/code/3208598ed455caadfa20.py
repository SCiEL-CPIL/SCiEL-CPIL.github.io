"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 85  # safety margin before 90s
    
    p = 1009
    n = 1008
    
    # Precompute modular inverses: inv[x] = x^(-1) mod p for x in [1, p-1]
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_len = 0
    best_start = 1
    L = 1
    
    for R in range(1, n + 1):
        # Self-inverse check: 1 and 1008 cannot be in any valid interval
        if R == 1 or R == 1008:
            L = R + 1
            continue
        
        invR = inv[R]
        
        # If invR is in the current window [L, R-1], we have a conflict
        if L <= invR < R:
            L = invR + 1
        
        # Window [L, R] is now valid by construction
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
        
        if time.time() > deadline:
            break
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
