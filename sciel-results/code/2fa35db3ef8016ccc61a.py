"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _primes_up_to(n):
    """Return list of primes <= n using sieve."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin within 60s limit
    
    gaps = P['gaps']
    n_pairs = len(gaps)
    
    # Compute the maximum gap size to know which primes we need
    max_gap = max(hi - lo for lo, hi in gaps)
    
    # Precompute all primes up to max_gap (these are the "forbidden" primes for each gap)
    forbidden_primes = _primes_up_to(max_gap)
    
    result = [None] * n_pairs
    
    # For each gap, find a witness m in (lo, hi) such that
    # the least prime factor of m is >= (hi - lo)
    # i.e., m is NOT divisible by any prime p < (hi - lo)
    
    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        gap = hi - lo
        # We need m such that no prime p < gap divides m
        # Forbidden primes for this gap: primes p where p < gap
        forbidden = [p for p in forbidden_primes if p < gap]
        
        # Search for m in (lo, hi)
        # Try all candidates; for each, check divisibility by forbidden primes
        found = False
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            valid = True
            for p in forbidden:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = m
                found = True
                break
    
    return result
# EVOLVE-BLOCK-END
