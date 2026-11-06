"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    p = 1009
    n = p - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 1
    best_len = 0
    
    L = 1
    
    for R in range(1, n + 1):
        if time.time() > deadline:
            break
            
        if R == 1 or R == 1008:
            # Self-inverse, cannot be in any valid interval
            L = R + 1
        else:
            inv_R = inv[R]
            # If inv_R is in the current window [L, R-1], we have a conflict
            # We must move L to be > inv_R
            if L <= inv_R < R:
                L = inv_R + 1
            
            # Now [L, R] is a valid interval
            current_len = R - L + 1
            if current_len > best_len:
                best_len = current_len
                best_start = L
                
    return [best_start, best_len]
# EVOLVE-BLOCK-END
