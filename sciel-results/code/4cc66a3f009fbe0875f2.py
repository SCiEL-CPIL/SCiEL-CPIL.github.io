"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    # inv[x] is the y in [1, p-1] such that x*y % p == 1
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    best_len = 0
    best_start = 1
    L = 1
    
    for R in range(1, n + 1):
        # Self-inverses 1 and 1008 cannot be in any valid interval
        if R == 1 or R == 1008:
            L = R + 1
            continue
            
        invR = inv[R]
        
        # Check if invR is in the current window [L, R-1]
        # If so, we have a conflict. We must move L past invR.
        if L <= invR < R:
            L = invR + 1
            
        # Now check if the window [L, R] is non-empty and valid
        # It is valid by construction:
        # 1. R is not self-inverse.
        # 2. invR is not in [L, R-1] because L > invR if it was there.
        # 3. No other new conflicts are introduced because [L, R-1] was valid 
        #    and [L, R] is a superset only by R, whose only partner is invR.
        #    Wait, is it possible that moving L creates a conflict?
        #    No, moving L to the right removes elements. It cannot create a pair 
        #    that wasn't there before. The only new element is R.
        
        if L <= R:
            curr_len = R - L + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = L
                
    return [best_start, best_len]
# EVOLVE-BLOCK-END
