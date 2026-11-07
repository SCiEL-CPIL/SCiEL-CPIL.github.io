"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes(limit):
    """Generate list of primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 second safety margin
    
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    small_primes = _sieve_primes(max_gap)  # primes up to max_gap
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        
        # Get primes strictly less than gap
        primes_lt_gap = [p for p in small_primes if p < gap]
        
        # Compute M = product of all primes < gap
        M = 1
        for p in primes_lt_gap:
            M *= p
        
        # For each residue r in [1, M-1] that is coprime to M (i.e., gcd(r,M)=1),
        # numbers m = k*M + r have no prime factor < gap.
        # We need m strictly between lo and hi.
        # So we need k*M + r > lo and k*M + r < hi
        # i.e., k > (lo - r) / M and k < (hi - r) / M
        
        found = False
        
        # Try small residues first (more likely to yield small k)
        for r in range(1, M):
            # Check gcd(r, M) == 1 (r has no prime factor < gap)
            has_small_factor = False
            for p in primes_lt_gap:
                if r % p == 0:
                    has_small_factor = True
                    break
            if has_small_factor:
                continue
            
            # Find valid k: lo < k*M + r < hi
            # k > (lo - r) / M and k < (hi - r) / M
            k_min = (lo - r) // M + 1  # smallest k such that k*M + r > lo
            k_max = (hi - r - 1) // M  # largest k such that k*M + r < hi
            
            if k_min <= k_max and k_min >= 0:
                # Use smallest valid k
                m = k_min * M + r
                if lo < m < hi:
                    result[i] = m
                    found = True
                    break
        
        # Fallback: brute force scan
        if not found:
            for m in range(lo + 1, hi):
                has_small_factor = False
                for p in primes_lt_gap:
                    if m % p == 0:
                        has_small_factor = True
                        break
                if not has_small_factor:
                    result[i] = m
                    found = True
                    break
        
        if time.time() > deadline:
            break
    
    # Ensure all entries are Python int or None
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))
    
    return final_result
# EVOLVE-BLOCK-END
