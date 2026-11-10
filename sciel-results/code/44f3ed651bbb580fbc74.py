import time
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Precompute modular inverses for 1 to N
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 1
    best_length = 0
    e = 0  # current right boundary
    
    for s in range(1, N + 1):
        if time.time() > deadline:
            break
        
        # If s is self-inverse, no valid interval can start at s (or contain s)
        if inv[s] == s:
            continue
            
        # Ensure e >= s
        if e < s:
            e = s
            
        # Expand e as far as possible
        while e < N:
            next_e = e + 1
            # Check if next_e can be added to [s, e]
            # Condition: inv[next_e] must not be in [s, next_e]
            inv_next = inv[next_e]
            if inv_next < s or inv_next > next_e:
                e = next_e
            else:
                break
                
        length = e - s + 1
        if length > best_length:
            best_length = length
            best_start = s
            
    return [best_start, best_length]
# EVOLVE-BLOCK-END
