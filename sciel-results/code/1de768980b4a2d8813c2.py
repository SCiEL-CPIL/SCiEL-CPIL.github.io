"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the longest contiguous interval [start, start+length-1] contained in [1, 1008]
    such that for every x, y in the interval (including x=y), x*y % 1009 != 1.
    
    Uses a two-pointer sliding window technique with O(N) inverse precomputation.
    """
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses in O(N) using the recurrence:
    # inv[i] = (MOD - MOD // i) * inv[MOD % i] % MOD
    inv = [0] * (MOD + 1)
    inv[1] = 1
    for i in range(2, MOD):
        inv[i] = (MOD - MOD // i) * inv[MOD % i] % MOD
        
    # Two-pointer sliding window
    # l is the start of the current valid window [l, r]
    l = 1
    best_start = 1
    best_length = 0
    
    for r in range(1, N + 1):
        # Self-inverse elements (1 and 1008) cannot be in any valid interval
        if r == 1 or r == N:
            l = r + 1
            continue
        
        inv_r = inv[r]
        
        # If inv_r is within the current window [l, r-1], we must move l
        # to exclude it. Since [l, r-1] was valid, any suffix of it is also valid.
        if l <= inv_r < r:
            l = inv_r + 1
        
        # The window [l, r] is now valid
        curr_len = r - l + 1
        if curr_len > best_length:
            best_length = curr_len
            best_start = l
            
    return [best_start, best_length]
# EVOLVE-BLOCK-END
