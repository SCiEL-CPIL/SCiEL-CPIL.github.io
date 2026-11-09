"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _primes_up_to(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

# Precompute small primes up to 101 (sqrt(10300) ~ 101.5)
_SMALL_PRIMES = _primes_up_to(101)

def _least_prime_factor(m):
    """Return the least prime factor of m, or m if m is prime."""
    for p in _SMALL_PRIMES:
        if p * p > m:
            break
        if m % p == 0:
            return p
    return m

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin from 90s limit
    
    n = len(P['gaps'])
    result = [None] * n
    
    # Precompute: for each gap, find primes strictly less than the gap size
    # A witness m needs lpf(m) >= gap, i.e., m is NOT divisible by any prime < gap
    # For gap g, forbidden primes are those < g
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() - start_time > DEADLINE:
            break
        gap = hi - lo
        # Forbidden primes: primes < gap
        # Since max gap is 16, forbidden primes are from {2,3,5,7,11,13} filtered by < gap
        forbidden = [p for p in _SMALL_PRIMES if p < gap]
        
        # Candidates: integers in (lo, hi)
        # Use numpy for vectorized check
        if hi - lo <= 1:
            # No integer strictly between
            continue
        
        candidates = np.arange(lo + 1, hi, dtype=np.int64)
        
        # Check if any candidate is divisible by any forbidden prime
        valid_mask = np.ones(len(candidates), dtype=bool)
        for p in forbidden:
            valid_mask &= (candidates % p != 0)
        
        if valid_mask.any():
            # Pick the first valid candidate
            idx = int(np.argmax(valid_mask))
            result[i] = int(candidates[idx])
    
    # Fallback: for any None entries, try a more exhaustive search with deadline
    for i, (lo, hi) in enumerate(P['gaps']):
        if result[i] is not None:
            continue
        if time.time() - start_time > DEADLINE:
            break
        gap = hi - lo
        for m in range(lo + 1, hi):
            lpf = _least_prime_factor(m)
            if lpf >= gap:
                result[i] = int(m)
                break
    
    # Ensure all are Python int or None
    return [int(x) if x is not None else None for x in result]
# EVOLVE-BLOCK-END
