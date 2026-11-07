import time
import math

_start_time = time.time()
_DEADLINE = 60.0

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _compute_lpf_range(start, end):
    """
    Compute Least Prime Factor for all integers in [start, end].
    Uses a segmented sieve approach or simple trial division if range is small.
    Given the small range (~300), simple trial division with primes up to sqrt(end) is sufficient.
    """
    lpf = [0] * (end - start + 1)
    # Generate primes up to sqrt(end)
    limit = int(math.isqrt(end)) + 1
    primes = []
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
            
    for i in range(start, end + 1):
        # Find LPF of i
        lpf_val = i # Default to i itself (prime)
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                lpf_val = p
                break
        lpf[i - start] = lpf_val
    return lpf

def construct():
    global _start_time, _DEADLINE
    if time.time() - _start_time > _DEADLINE:
        return [None] * len(P['gaps'])
        
    # Determine range
    min_val = min(lo for lo, hi in P['gaps'])
    max_val = max(hi for lo, hi in P['gaps'])
    
    # Compute LPF for the range
    lpf_arr = _compute_lpf_range(min_val, max_val)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() - _start_time > _DEADLINE:
            break
            
        gap = hi - lo
        if gap <= 1:
            continue
            
        witness = None
        # Search for witness in (lo, hi)
        for m in range(lo + 1, hi):
            # Check if LPF(m) >= gap
            if lpf_arr[m - min_val] >= gap:
                witness = m
                break
                
        if witness is not None:
            result[i] = int(witness)
            
    return [int(x) if x is not None else None for x in result]
