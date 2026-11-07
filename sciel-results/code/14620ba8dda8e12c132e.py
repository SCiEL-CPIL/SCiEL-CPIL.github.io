# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _least_prime_factor(m, small_primes):
    """
    Return the least prime factor of m, or m itself if m is prime.
    small_primes should contain all primes up to sqrt(m) for efficiency.
    """
    for p in small_primes:
        if p * p > m:
            break
        if m % p == 0:
            return p
    # If we get here, m is prime (or 1, but m >= 2 in our use case)
    return m

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n
    
    # Precompute small primes up to sqrt(max possible m)
    max_m = max(hi for _, hi in gaps)
    sqrt_max = isqrt(max_m) + 1
    small_primes = _sieve_primes(sqrt_max)
    
    # For each gap, find a witness
    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        
        gap_size = hi - lo
        
        # We need m in (lo, hi) such that least_prime_factor(m) >= gap_size
        # This means m has no prime factor less than gap_size
        # i.e., m is either prime, or all its prime factors are >= gap_size
        
        # Strategy: try numbers from the middle outward, or just scan
        # For small gaps, scanning is fine. For larger gaps, we need to be smarter.
        
        # Check candidates: prefer numbers that are prime or have large least prime factor
        # A number m has LPF >= g iff m is not divisible by any prime < g
        
        # Generate primes less than gap_size for filtering
        primes_below_gap = [p for p in small_primes if p < gap_size]
        
        found = False
        # Try candidates in order: check from lo+1 to hi-1
        # But we can be smarter: skip numbers divisible by any prime < gap_size
        
        # For efficiency, if gap_size is small, most numbers work
        # If gap_size is large, we need numbers with no small factors
        
        # Let's just scan but with early termination
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            
            # Check if m has any prime factor < gap_size
            has_small_factor = False
            for p in primes_below_gap:
                if p * p > m:
                    break
                if m % p == 0:
                    has_small_factor = True
                    break
            
            if not has_small_factor:
                # Verify: least prime factor should be >= gap_size
                lpf = _least_prime_factor(m, small_primes)
                if lpf >= gap_size:
                    result[i] = m
                    found = True
                    break
        
        if not found:
            # Try a more targeted approach: look for primes in the range
            # or numbers with large least prime factor
            # For now, leave as None
            pass
    
    # Ensure all entries are Python int or None
    for i in range(n):
        if result[i] is not None:
            result[i] = int(result[i])
    
    return result
# EVOLVE-BLOCK-END
