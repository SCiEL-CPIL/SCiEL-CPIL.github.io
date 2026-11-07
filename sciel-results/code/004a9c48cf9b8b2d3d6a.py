"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    from math import isqrt
    import numpy as np
    
    # Determine the range we need to cover
    min_val = min(lo for lo, hi in P['gaps'])
    max_val = max(hi for lo, hi in P['gaps'])
    
    # We need to find least prime factor for numbers in [min_val+1, max_val-1]
    # A number n has least prime factor >= g if it has no prime factor < g
    # i.e., n is not divisible by any prime < g
    
    # Strategy: For each gap with size g, we need a number m in (lo, hi) 
    # such that m has no prime factor < g.
    # This means m is either prime, or a product of primes all >= g.
    
    # Precompute primes up to sqrt(max_val) for factorization
    sqrt_max = isqrt(max_val)
    # Sieve of Eratosthenes to get primes up to sqrt_max
    sieve = np.ones(sqrt_max + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(sqrt_max) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    small_primes = [int(p) for p in np.where(sieve)[0]]
    
    # For each pair, search for a witness
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # We need m in (lo, hi) with least prime factor >= gap
        # This means m is not divisible by any prime < gap
        # Precompute which primes < gap we need to check
        primes_to_check = [p for p in small_primes if p < gap]
        
        found = False
        for m in range(lo + 1, hi):
            # Check if m has any prime factor < gap
            has_small_factor = False
            for p in primes_to_check:
                if m % p == 0:
                    has_small_factor = True
                    break
            if not has_small_factor:
                # m has no prime factor < gap, so least prime factor >= gap
                result[i] = m
                found = True
                break
    
    return result
# EVOLVE-BLOCK-END
