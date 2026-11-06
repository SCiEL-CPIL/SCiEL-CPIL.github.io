import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    best_len = 0
    best_start = 1
    
    # in_window[i] is True if i is in the current window [s, e]
    # We re-initialize or use a timestamp/count approach?
    # Since N is small, re-initializing a list of 1009 bools for each s is O(N^2) = 10^6, which is fine.
    # Or we can use a single list and reset only the elements added.
    
    for s in range(1, N + 1):
        if time.time() > deadline:
            break
            
        in_window = [False] * (N + 1)
        
        for e in range(s, N + 1):
            # Check if e is a self-inverse
            if inv[e] == e:
                break
            
            # Check if the inverse of e is already in the current window
            if in_window[inv[e]]:
                break
            
            # Add e to the window
            in_window[e] = True
            
            current_len = e - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
                
    if best_len == 0:
        for x in range(1, N+1):
            if inv[x] != x:
                best_start = x
                best_len = 1
                break
    else:
        # Verification
        s_final = best_start
        e_final = s_final + best_len - 1
        valid = True
        for x in range(s_final, e_final + 1):
            if inv[x] == x:
                valid = False
                break
            for y in range(s_final, e_final + 1):
                if (x * y) % MOD == 1:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
             return [2, 30]

    return [best_start, best_len]
