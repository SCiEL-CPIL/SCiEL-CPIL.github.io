import time

def construct():
    p = 1009
    n = 1008
    
    # Deadline guard (safety, though N=1008 is tiny)
    deadline = time.time() + 85.0
    
    # Precompute inverses
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    best_len = 0
    best_start = 1
    L = 1
    
    for R in range(1, n + 1):
        if time.time() > deadline:
            break
            
        # Self-inverse residues (1 and p-1) cannot be in any valid interval
        if R == 1 or R == p - 1:
            L = R + 1
            continue
        
        invR = inv[R]
        
        # If the inverse of R is within the current window [L, R-1],
        # we must move L to exclude it.
        if L <= invR < R:
            L = invR + 1
            
        # The window [L, R] is now valid.
        curr_len = R - L + 1
        if curr_len > best_len:
            best_len = curr_len
            best_start = L
            
    return [best_start, best_len]
