# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _gen_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(limit + 1) if sieve[i]]

def _least_prime_factor(m, small_primes):
    """Return least prime factor of m, or m if m is prime."""
    for p in small_primes:
        if p * p > m:
            return m
        if m % p == 0:
            return p
    return m

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safe margin before 60s kill
    
    # Precompute primes up to sqrt(max value in gaps)
    small_primes = _gen_primes(110)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        found = False
        
        # Generate all candidates in the gap
        candidates = list(range(lo + 1, hi))
        
        # For a given gap g, we need m not divisible by any prime < g
        primes_below_gap = [p for p in small_primes if p < gap]
        
        # Score each candidate: count how many primes below gap divide it
        scored = []
        for m in candidates:
            div_count = 0
            for p in primes_below_gap:
                if m % p == 0:
                    div_count += 1
            scored.append((div_count, m))
        
        # Sort by number of bad divisors (fewer is better)
        scored.sort(key=lambda x: (x[0], x[1]))
        
        # Try candidates in order of preference
        for _, m in scored:
            lpf = _least_prime_factor(m, small_primes)
            if lpf >= gap:
                result[i] = int(m)
                found = True
                break
        
        # Second pass: try all candidates linearly
        if not found:
            for m in candidates:
                lpf = _least_prime_factor(m, small_primes)
                if lpf >= gap:
                    result[i] = int(m)
                    found = True
                    break
    
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))
    
    return final_result

# EVOLVE-BLOCK-END
