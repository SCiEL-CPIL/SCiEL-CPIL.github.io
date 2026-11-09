import time

def construct():
    start_time = time.time()
    LIMIT = 1009
    MAX_VAL = 1008
    
    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)
        
    best_len = 0
    best_start = 2
    
    L = 2
    # R goes from 2 to 1007
    for R in range(2, 1008):
        # Check if inv[R] is in [L, R-1]
        if inv[R] >= L:
            L = inv[R] + 1
            
        length = R - L + 1
        if length > best_len:
            best_len = length
            best_start = L
            
        # Time check (though this is very fast)
        if time.time() - start_time > 80:
            break
            
    return [best_start, best_len]
