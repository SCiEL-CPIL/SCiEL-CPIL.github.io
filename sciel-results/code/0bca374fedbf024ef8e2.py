"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find integer n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy:
    1. Use a modified sieve to compute the largest prime factor (LPF) for every integer up to 1,000,001.
    2. For each n in [10000, 1000000], compute LPF(n * (n+1)) = max(LPF(n), LPF(n+1)).
    3. Find the n that minimizes this value.
    
    We use an array `lpf` where lpf[i] is the largest prime factor of i.
    We compute this efficiently by iterating through primes in increasing order and updating multiples.
    """
    MAX_N = 1000000
    MAX_VAL = MAX_N + 1  # We need LPF up to 1000001
    
    # Initialize LPF array
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)
    lpf[0] = 0
    lpf[1] = 1
    
    # Sieve for largest prime factor
    # For p from 2 to MAX_VAL:
    #   if lpf[p] == 0: p is prime
    #   for multiple m of p: lpf[m] = p
    # Since we iterate p in increasing order, the last assignment to lpf[m] will be from the largest prime factor of m.
    
    for p in range(2, MAX_VAL + 1):
        if lpf[p] == 0:  # p is prime
            # Use vectorized assignment for speed
            lpf[p:MAX_VAL+1:p] = p
    
    # Now find the n in [10000, 1000000] that minimizes max(lpf[n], lpf[n+1])
    # Vectorized computation of the max for all n at once
    n_range = np.arange(10000, MAX_N + 1)
    lpf_n = lpf[n_range]
    lpf_n1 = lpf[n_range + 1]
    vals = np.maximum(lpf_n, lpf_n1)
    
    best_idx = np.argmin(vals)
    best_n = int(n_range[best_idx])
    
    return best_n
# EVOLVE-BLOCK-END
