"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes_up_to(n):
    """Return list of primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _least_prime_factor(m, primes_list):
    """Return least prime factor of m, or m itself if prime."""
    for p in primes_list:
        if p * p > m:
            break
        if m % p == 0:
            return p
    return m

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    primes_up_to_max = _sieve_primes_up_to(max_gap)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Primes strictly less than gap that would disqualify a candidate
        disqualifying = [p for p in primes_up_to_max if p < gap]
        
        if not disqualifying:
            # No prime < gap exists, so any m has LPF >= gap (or m is prime with LPF=m>=gap)
            # Actually if no prime < gap, then LPF(m) >= gap for all m > 1
            result[i] = lo + 1
            continue
        
        # Check each candidate in (lo, hi)
        found = False
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            # Check if any disqualifying prime divides m
            valid = True
            for p in disqualifying:
                if p * p > m:
                    break
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = m
                found = True
                break
    
    # Ensure Python int types
    final_result = []
    for v in result:
        if v is None:
            final_result.append(None)
        else:
            final_result.append(int(v))
    
    return final_result
# EVOLVE-BLOCK-END
