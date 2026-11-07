"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(range(i*i, limit+1, i)))
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Precompute primes up to a sufficient limit.
    primes = _sieve_primes(100)
    
    # Precompute small_primes for each possible gap size (up to 30)
    small_primes_cache = {}
    for g in range(2, 31):
        small_primes_cache[g] = [p for p in primes if p < g]
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        small_primes = small_primes_cache.get(gap, [p for p in primes if p < gap])
        
        # Search for a witness by scanning all integers in the gap
        # Try from both ends to potentially find witnesses faster in tight gaps
        found = False
        for m in range(lo + 1, hi):
            if not any(m % p == 0 for p in small_primes):
                result[i] = int(m)
                found = True
                break
        
        if not found:
            # Also try from the high end (though we already scanned all, this is redundant)
            # but ensures we don't miss anything due to any subtle issue
            for m in range(hi - 1, lo, -1):
                if not any(m % p == 0 for p in small_primes):
                    result[i] = int(m)
                    found = True
                    break
    
    # Ensure all non-None values are Python int
    final_result = []
    for v in result:
        if v is None:
            final_result.append(None)
        else:
            final_result.append(int(v))
    
    return final_result
# EVOLVE-BLOCK-END
