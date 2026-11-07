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
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Precompute primes up to 200 (sqrt(10289) ~ 101.4)
    small_primes = _gen_primes(200)
    
    result = [None] * len(P['gaps'])
    
    # For each gap, find a witness m strictly between lo and hi
    # such that LPF(m) >= gap_size
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        
        # Find m in (lo, hi) such that LPF(m) >= gap
        found = False
        
        # Strategy: try candidates that are likely to have large LPF
        # Numbers coprime to small primes are more likely to have large LPF
        # Generate candidates in order of increasing "small prime divisibility"
        
        # First, generate all candidates in the gap
        candidates = list(range(lo + 1, hi))
        
        # Sort candidates by how many small primes divide them (ascending)
        # This prioritizes numbers with fewer small prime factors
        def count_small_prime_divisors(m):
            count = 0
            for p in small_primes:
                if p * p > m:
                    break
                if m % p == 0:
                    count += 1
            return count
        
        # Sort by number of small prime divisors, then by the value of the smallest dividing prime
        candidates.sort(key=lambda m: (count_small_prime_divisors(m), _least_prime_factor(m, small_primes) if _least_prime_factor(m, small_primes) < m else m))
        
        for m in candidates:
            if time.time() > deadline:
                break
            lpf = _least_prime_factor(m, small_primes)
            if lpf >= gap:
                result[i] = int(m)
                found = True
                break
        
        # If not found with sorted order, try linear scan as fallback
        if not found:
            for m in range(lo + 1, hi):
                if time.time() > deadline:
                    break
                lpf = _least_prime_factor(m, small_primes)
                if lpf >= gap:
                    result[i] = int(m)
                    found = True
                    break
    
    # Verify and convert to ensure Python int types
    final_result = []
    for i, val in enumerate(result):
        if val is None:
            final_result.append(None)
        else:
            lo, hi = P['gaps'][i]
            gap = hi - lo
            # Verify the witness is valid
            if lo < val < hi and _least_prime_factor(val, small_primes) >= gap:
                final_result.append(int(val))
            else:
                final_result.append(None)
    
    return final_result

# EVOLVE-BLOCK-END
