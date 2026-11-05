import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 1
    best_len = 0
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Monotonic two-pointer: for each R, find the smallest L such that [L, R] is valid
    L = 1
    for R in range(1, N + 1):
        if time.time() > deadline:
            break
        
        # Adding R: check if inv[R] falls in [L, R]
        if inv[R] >= L and inv[R] <= R:
            L = inv[R] + 1
        
        # [L, R] is now valid
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
    
    return [best_start, best_len]
