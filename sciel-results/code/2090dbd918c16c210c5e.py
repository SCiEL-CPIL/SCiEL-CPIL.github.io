"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    # Precompute primes up to 100 (sqrt(10300) is ~101, but we only need primes < gap, max gap 28)
    # So primes up to 28 are sufficient for the "small factor" check.
    # However, to be safe and general, we sieve up to 100.
    def sieve(n):
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_p[i]:
                for j in range(i * i, n + 1, i):
                    is_p[j] = False
        return [i for i in range(2, n + 1) if is_p[i]]

    primes_small = sieve(100)
    
    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        # Candidates are m in (lo, hi)
        # We need LPF(m) >= gap
        # This is equivalent to: m is not divisible by any prime p < gap
        
        # Get relevant primes
        check_primes = [p for p in primes_small if p < gap]
        
        # Use a boolean sieve for the interval
        # Size of interval is 'gap'
        # Indices 0 to gap-1 correspond to lo+1 to hi-1
        
        valid = [True] * gap
        
        # Mark multiples of each check_prime as invalid
        for p in check_primes:
            # Find first multiple of p in (lo, hi)
            # m = k * p
            # lo < k * p < hi
            # ceil(lo/p) < k < hi/p
            # k_min = floor(lo/p) + 1
            k_min = lo // p + 1
            # Start marking from index corresponding to k_min * p
            # Index j corresponds to m = lo + 1 + j
            # m = k_min * p
            # j = k_min * p - (lo + 1)
            start_idx = k_min * p - (lo + 1)
            
            if start_idx < 0:
                start_idx = 0
                
            for j in range(start_idx, gap, p):
                if j < gap:
                    valid[j] = False
        
        # Find the first valid index
        found = False
        for j in range(gap):
            if valid[j]:
                m = lo + 1 + j
                result[i] = int(m)
                found = True
                break
        
        # If not found, result[i] remains None

    return result
# EVOLVE-BLOCK-END
