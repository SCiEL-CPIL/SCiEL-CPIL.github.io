import time

def construct():
    MOD = 1009
    N = 1008
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s out of 60s
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        inv[i] = pow(i, MOD - 2, MOD)
        
    # min_R_at_L[L] = min { R_x | L_x == L }
    # Note: For each L, there is at most one pair {L, R} where L is the min.
    # So min is just the value itself.
    min_R_at_L = [float('inf')] * (MOD + 2)
    
    for x in range(1, MOD):
        y = inv[x]
        L = min(x, y)
        R = max(x, y)
        if R < min_R_at_L[L]:
            min_R_at_L[L] = R
            
    # Compute suffix minimums M[s] = min { min_R_at_L[k] | k >= s }
    M = [float('inf')] * (MOD + 2)
    curr_min = float('inf')
    # Iterate from 1008 down to 1
    for s in range(MOD - 1, 0, -1):
        if min_R_at_L[s] < curr_min:
            curr_min = min_R_at_L[s]
        M[s] = curr_min
        
    # Find max length
    best_len = 0
    best_start = 1
    
    for s in range(1, MOD):
        # Check time
        if time.time() > deadline:
            break
            
        max_e = M[s] - 1
        if max_e > MOD - 1:
            max_e = MOD - 1
            
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                
    return [best_start, best_len]
