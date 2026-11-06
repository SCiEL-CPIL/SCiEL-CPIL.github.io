import time

def construct():
    p = 1009
    n = 1008
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin

    # Precompute modular inverses
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Precompute self-inverse flags (x * x % p == 1)
    # In a prime field, only 1 and p-1 are self-inverse
    is_self_inv = [False] * p
    for x in range(1, p):
        if (x * x) % p == 1:
            is_self_inv[x] = True

    best_start = 1
    best_len = 0
    
    # Two-pointer / Sliding window style search
    # For each start s, find the maximum end e such that [s, e] is valid.
    for s in range(1, n + 1):
        if time.time() > deadline:
            break
            
        in_window = [False] * p
        max_len_for_s = 0
        
        for e in range(s, n + 1):
            x = e
            
            # Check for self-inverse (diagonal conflict)
            if is_self_inv[x]:
                break
            
            y = inv[x]
            
            # Check if the inverse is already in the window
            if in_window[y]:
                break
            
            in_window[x] = True
            max_len_for_s = e - s + 1
        
        if max_len_for_s > best_len:
            best_len = max_len_for_s
            best_start = s
            
    # If no valid interval found (should not happen as length 1 is always valid for x!=1,1008)
    if best_len == 0:
        # Find any x != 1, 1008
        for x in range(1, n + 1):
            if x != 1 and x != 1008:
                return [x, 1]
        return [2, 1] # Fallback

    return [best_start, best_len]
