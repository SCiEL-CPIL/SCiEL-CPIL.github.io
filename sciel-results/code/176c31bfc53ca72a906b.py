import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds safety margin
    
    N = 1008
    MOD = 1009
    
    # Precompute modular inverses
    # inv[x] is the inverse of x modulo 1009
    inv = [0] * (MOD)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    best_start = 1
    best_len = 0
    
    # Iterate over all possible start positions
    for L in range(1, N + 1):
        # Check time
        if time.time() > deadline:
            break
            
        current_set = set()
        # Iterate over end positions
        for R in range(L, N + 1):
            # Check if R is self-inverse (R * R % MOD == 1)
            if inv[R] == R:
                break
            
            # Check if inverse of R is already in the interval [L, R-1]
            if inv[R] in current_set:
                break
            
            # R is valid to add
            current_set.add(R)
            
            current_len = R - L + 1
            if current_len > best_len:
                best_len = current_len
                best_start = L
                
    return [best_start, best_len]
