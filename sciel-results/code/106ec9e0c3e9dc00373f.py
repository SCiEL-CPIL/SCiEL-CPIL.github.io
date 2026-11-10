def construct():
    MOD = 1009
    N = MOD - 1
    
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)
        
    best_start = 1
    best_length = 0
    L = 1
    for R in range(1, N + 1):
        # If inv[R] is in [L, R], we must move L past inv[R]
        if L <= inv[R] <= R:
            L = inv[R] + 1
            
        length = R - L + 1
        if length > best_length:
            best_length = length
            best_start = L
            
    return [best_start, best_length]
