"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    n = p - 1  # 1008
    
    # Precompute modular inverses and forbidden pairs
    # A pair {x, y} is forbidden if x*y % p == 1.
    # Let L = min(x, y), R = max(x, y).
    # If an interval [s, e] contains both L and R, it's invalid.
    # This is equivalent to s <= L and e >= R.
    # For a fixed s, we need e < R for all pairs with L >= s.
    # So max e = min(R_i for all i where L_i >= s) - 1.
    
    # Initialize array to hold the minimum R for each starting L index
    # We only care about L in [1, n]
    INF = 10**9
    min_R_for_L = [INF] * (n + 2)
    
    # Iterate x from 1 to n
    for x in range(1, n + 1):
        y = pow(x, p - 2, p)
        L = min(x, y)
        R = max(x, y)
        # Update the minimum R for this L
        if R < min_R_for_L[L]:
            min_R_for_L[L] = R
            
    # Compute suffix minimums
    # suffix_min[s] = min(min_R_for_L[i] for i in [s, n])
    suffix_min = [INF] * (n + 2)
    # suffix_min[n+1] is INF (empty set)
    for s in range(n, 0, -1):
        suffix_min[s] = min(min_R_for_L[s], suffix_min[s + 1])
        
    best_len = 0
    best_start = 1
    
    # Check all possible starts s
    for s in range(1, n + 1):
        max_R = suffix_min[s]
        # The condition is e < max_R. So max e = max_R - 1.
        max_e = max_R - 1
        
        # The interval must be non-empty and within bounds
        # e must be at least s
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                
    return [best_start, best_len]
# EVOLVE-BLOCK-END
