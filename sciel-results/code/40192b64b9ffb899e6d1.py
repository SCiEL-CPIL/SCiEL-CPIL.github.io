"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    MOD = 1009
    N = 1008
    
    # Compute modular inverse for each x in 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    # Valid range for interval elements is [2, 1007] because 1 and 1008 are self-inverse
    lo_bound = 2
    hi_bound = 1007
    
    best_start = 2
    best_length = 0
    
    for s in range(lo_bound, hi_bound + 1):
        if time.time() > DEADLINE:
            break
            
        e = s
        # Expand e as far as possible
        while e < hi_bound:
            next_val = e + 1
            inv_next = inv[next_val]
            
            # Check if adding next_val creates a forbidden pair within [s, next_val]
            # A pair (x, next_val) is forbidden if x == inv[next_val]
            # x must be in [s, e] (since next_val is new)
            # If inv[next_val] is in [s, e], we have a conflict.
            if inv_next >= s and inv_next <= e:
                break
            else:
                e = next_val
                
        current_length = e - s + 1
        if current_length > best_length:
            best_length = current_length
            best_start = s
            
    return [best_start, best_length]
# EVOLVE-BLOCK-END
