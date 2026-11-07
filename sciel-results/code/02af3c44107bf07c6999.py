"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _get_primes_below(n):
    """Return a list of primes strictly less than n."""
    if n <= 2:
        return []
    sieve = [True] * n
    sieve[0] = False
    sieve[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            for j in range(i*i, n, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    result = [None] * len(P['gaps'])
    # Precompute primes for max gap
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    all_primes = _get_primes_below(max_gap)
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        if gap <= 2:
            continue # No integer strictly between
        
        # Primes strictly less than gap
        bad_primes = [p for p in all_primes if p < gap]
        
        found = None
        for m in range(lo + 1, hi):
            # Check if m is divisible by any bad prime
            is_valid = True
            for p in bad_primes:
                if m % p == 0:
                    is_valid = False
                    break
            if is_valid:
                found = m
                break
        
        if found is not None:
            result[i] = int(found)
            
    return result
# EVOLVE-BLOCK-END
