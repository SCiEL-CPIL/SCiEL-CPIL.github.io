"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    from math import isqrt
    
    # Precompute primes up to max gap
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    
    # Sieve to get all primes up to max_gap
    sieve = [True] * (max_gap + 1)
    if max_gap >= 0:
        sieve[0] = False
    if max_gap >= 1:
        sieve[1] = False
    for i in range(2, isqrt(max_gap) + 1):
        if sieve[i]:
            for j in range(i*i, max_gap + 1, i):
                sieve[j] = False
    all_primes = [i for i in range(2, max_gap + 1) if sieve[i]]
    
    # Precompute forbidden primes for each possible gap size
    # forbidden[g] contains primes p such that p < g
    forbidden_cache = []
    for g in range(max_gap + 1):
        forbidden_cache.append([p for p in all_primes if p < g])
        
    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        forbidden = forbidden_cache[gap]
        for m in range(lo + 1, hi):
            # Check if m is divisible by any forbidden prime
            valid = True
            for p in forbidden:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = m
                break
                
    return result
# EVOLVE-BLOCK-END
