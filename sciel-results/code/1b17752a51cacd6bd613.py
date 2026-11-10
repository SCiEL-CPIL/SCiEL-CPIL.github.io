import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    MOD = 1009
    
    # Precompute inverses
    inv = [0] * MOD
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    best_start = 0
    best_length = 0
    
    # Iterate over all possible starts
    for s in range(1, MOD):
        if time.time() > deadline:
            break
            
        # Check if s itself is valid (s * s != 1 mod MOD)
        if inv[s] == s:
            continue
            
        # Greedily extend the interval
        e = s
        # We can extend as long as the next element's inverse is not in [s, e]
        while e < MOD - 1:
            if time.time() > deadline:
                break
            new_e = e + 1
            # Check if new_e can be added
            # new_e is valid to add if inv[new_e] is NOT in [s, e]
            if inv[new_e] < s or inv[new_e] > e:
                e = new_e
            else:
                break
                
        length = e - s + 1
        if length > best_length:
            best_length = length
            best_start = s
            
    # If no valid interval found (should not happen, e.g. [2,2] is valid), return a default
    if best_length == 0:
        return [2, 1]
        
    return [best_start, best_length]
