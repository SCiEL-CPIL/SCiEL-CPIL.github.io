"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s kill
    
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Sweep-line approach:
    # For each position x, the inverse pair is (x, inv[x]).
    # An interval [s, e] is invalid if it contains both x and inv[x] for any x.
    # 
    # For each x, define L_x = min(x, inv[x]) and R_x = max(x, inv[x]).
    # The interval [s, e] is valid iff for no x do we have L_x >= s and R_x <= e.
    # Equivalently, for a fixed start s, the maximum valid end is:
    #   e_max(s) = min{ R_x - 1 : L_x >= s }
    # (if no such x exists, e_max(s) = N)
    #
    # We can precompute this efficiently using a suffix minimum.
    
    # For each possible left endpoint L, find the minimum R among all pairs with that L
    min_R_at_L = [float('inf')] * (MOD + 2)
    
    for x in range(1, MOD):
        y = inv[x]
        L = min(x, y)
        R = max(x, y)
        if R < min_R_at_L[L]:
            min_R_at_L[L] = R
    
    # Compute suffix minimum: min_R_from[s] = min{ min_R_at_L[L] : L >= s }
    # This gives us the minimum R_x for any pair whose left endpoint is at least s.
    min_R_from = [float('inf')] * (MOD + 2)
    curr_min = float('inf')
    for s in range(MOD, 0, -1):
        if min_R_at_L[s] < curr_min:
            curr_min = min_R_at_L[s]
        min_R_from[s] = curr_min
    
    # Now find the longest valid interval
    best_start = 2
    best_len = 0
    
    for s in range(1, MOD):
        if time.time() > deadline:
            break
        
        # Maximum valid end for start s
        max_e = min_R_from[s] - 1
        if max_e > N:
            max_e = N
        
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    if best_len == 0:
        # Fallback: find any single valid element
        # 1 and 1008 are self-inverse, so they're invalid
        # Any other single element is valid
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
