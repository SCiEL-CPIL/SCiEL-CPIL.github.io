import time

def construct():
    MOD = 1009
    N = 1008
    start_time = time.time()
    deadline = 80  # safety margin
    
    # Precompute inverses
    inv = [0] * (MOD)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 2
    best_len = 0
    
    for L in range(2, 1008):
        if time.time() - start_time > deadline:
            break
        # Find max R such that for all x in [L, R], inv(x) is not in [L, R]
        R_max = 1007
        for x in range(L, 1008):
            ix = inv[x]
            if ix >= L:
                # R must be < ix
                if ix - 1 < R_max:
                    R_max = ix - 1
                if R_max < L:
                    break
        if R_max >= L:
            length = R_max - L + 1
            if length > best_len:
                best_len = length
                best_start = L
    
    return [best_start, best_len]
