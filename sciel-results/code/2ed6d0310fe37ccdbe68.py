import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = 1008
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin for 60s limit
    
    # Precompute modular inverses
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    best_len = 0
    best_start = 1
    L = 1
    
    for R in range(1, n + 1):
        if time.time() > deadline:
            break
            
        # Self-inverses 1 and 1008 cannot be in any valid interval
        if R == 1 or R == 1008:
            L = R + 1
            continue
            
        invR = inv[R]
        
        # Check if invR is in the current window [L, R-1]
        if L <= invR < R:
            L = invR + 1
            
        if L <= R:
            curr_len = R - L + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = L
                
    return [best_start, best_len]
# EVOLVE-BLOCK-END
