import time

def construct():
    # Start time for safety
    start_time = time.time()
    deadline = 60.0  # Seconds

    p = 1009
    n = 1008
    
    # Precompute modular inverses
    # inv[x] is the modular inverse of x modulo p
    inv = [0] * (p)
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
        
    # Two-pointer algorithm to find longest valid interval
    # Valid interval [L, R] means for all x in [L, R], inv[x] is not in [L, R]
    # This is equivalent to: for all y in [L, R], inv[y] < L or inv[y] > R
    
    max_len = 0
    best_start = 1
    
    L = 1
    R = 0
    
    while L <= n:
        # If we've hit the deadline, break out (unlikely)
        if time.time() > deadline - 5.0:
            break
            
        # Try to extend R as far as possible
        # Condition to add R+1: inv[R+1] must not be in [L, R]
        while R < n:
            next_y = R + 1
            # Check if inv[next_y] is outside [L, R]
            if inv[next_y] < L or inv[next_y] > R:
                R = next_y
            else:
                break
        
        # Update maximum length
        if R >= L:
            current_len = R - L + 1
            if current_len > max_len:
                max_len = current_len
                best_start = L
        
        L += 1
        
    # Ensure we return a valid non-empty interval
    # If max_len is 0 (should not happen), return a default valid one
    if max_len == 0:
        # Find any valid single element
        # 2 is valid because inv[2] = 505, which is not 2
        return [2, 1]
        
    return [best_start, max_len]
