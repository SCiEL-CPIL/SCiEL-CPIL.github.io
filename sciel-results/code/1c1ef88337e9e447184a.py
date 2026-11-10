"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find the longest interval [start, start+length-1] within [1, 1008]
    such that for every x, y in the interval (including x=y), x*y % 1009 != 1.
    
    Approach:
    1. Precompute modular inverses for 1..1008 modulo 1009.
    2. For each potential start L (1..1008), determine the maximum valid end R.
       An interval [L, R] is valid iff for all x in [L, R], inv(x) is not in [L, R].
       This implies for all x in [L, R], inv(x) < L or inv(x) > R.
       
       Let S_L = { x in [L, 1008] | inv(x) >= L }.
       If S_L is empty, we can extend R to 1008.
       If S_L is not empty, let M_L = min(inv(x) for x in S_L).
       Then we must have R < M_L, so max R is M_L - 1.
       
    3. Iterate L from 1 to 1008, compute max R, calculate length, and track the maximum.
    """
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
        
    best_start = 1
    best_len = 0
    
    # For each L, find the maximum valid R
    for L in range(1, n + 1):
        min_inv = n + 1  # Initialize to infinity (n+1 = 1009)
        
        # Find min inv(x) for x >= L such that inv(x) >= L
        # We iterate x from L to n
        for x in range(L, n + 1):
            if inv[x] >= L:
                if inv[x] < min_inv:
                    min_inv = inv[x]
                    
        if min_inv == n + 1:
            R = n
        else:
            R = min_inv - 1
            
        if R >= L:
            length = R - L + 1
            if length > best_len:
                best_len = length
                best_start = L
                
    return [best_start, best_len]
# EVOLVE-BLOCK-END
