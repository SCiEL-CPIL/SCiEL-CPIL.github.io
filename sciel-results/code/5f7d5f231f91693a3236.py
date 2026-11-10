"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    MOD = 1009
    
    # Precompute modular inverses for 1 to 1008
    # inv[x] = x^(-1) mod 1009
    inv = [0] * MOD
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    best_start = 2
    best_length = 0
    
    # Sliding window approach
    # L is the start of the current valid window [L, R]
    # We iterate R from 2 to 1007
    # 1 and 1008 are self-inverses, so they cannot be in the interval
    L = 2
    
    for R in range(2, 1008):
        if time.time() > deadline:
            break
            
        # Check if adding R to [L, R-1] creates a conflict
        # Conflict exists if inv[R] is in [L, R]
        # Since we are adding R, the new pair is (R, inv[R]).
        # If inv[R] is in [L, R], then L <= inv[R] <= R.
        # If this happens, we must move L to inv[R] + 1 to exclude inv[R] from the window.
        # Note: Moving L to the right only removes elements, so it cannot create new conflicts
        # for existing elements. The new element R's inverse is now < L, so it's safe.
        
        if inv[R] >= L and inv[R] <= R:
            L = inv[R] + 1
            
        # Now [L, R] is valid
        current_length = R - L + 1
        if current_length > best_length:
            best_length = current_length
            best_start = L
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
