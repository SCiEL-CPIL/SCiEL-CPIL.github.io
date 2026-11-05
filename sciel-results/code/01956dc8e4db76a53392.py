"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    # inv[x] is the number y in [1, 1008] such that x*y % 1009 == 1
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    best_start = 2
    best_len = 0
    start_time = time.time()
    
    # We iterate over all possible starting positions S
    # S must be in [1, 1008]
    # If S is 1 or 1008, the interval contains a self-inverse, so it's invalid.
    # So we can restrict S to [2, 1007]
    
    for S in range(2, MOD - 1):
        # Check time limit
        if time.time() - start_time > 50:
            break
            
        E = S - 1
        # Extend E as far as possible
        while E < MOD - 1: # E can go up to 1008
            new_E = E + 1
            
            # If new_E is 1008, we cannot include it (self-inverse)
            # new_E cannot be 1 because S >= 2 and E >= S-1 >= 1, so new_E >= 2
            if new_E == 1008:
                break
                
            # Check if adding new_E creates a conflict
            # A conflict exists if inv(new_E) is already in [S, E]
            inv_new_E = inv[new_E]
            
            # Check if inv_new_E is in the current interval [S, E]
            if S <= inv_new_E <= E:
                break
            else:
                E = new_E
        
        # If we formed a valid interval [S, E]
        if E >= S:
            current_len = E - S + 1
            if current_len > best_len:
                best_len = current_len
                best_start = S
                
    return [best_start, best_len]
# EVOLVE-BLOCK-END
