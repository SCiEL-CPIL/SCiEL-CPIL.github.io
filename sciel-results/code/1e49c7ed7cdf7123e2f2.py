import time
import numpy as np

_start_time = time.time()
_DEADLINE = 85.0

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _compute_lpf(limit):
    """
    Compute least prime factor for all numbers up to limit.
    lpf[n] is the smallest prime factor of n. If n is prime, lpf[n] = n.
    Uses a linear-ish sieve approach with NumPy vectorization.
    """
    lpf = np.zeros(limit + 1, dtype=np.int64)
    # Initialize: lpf[i] = 0 means not yet assigned
    # For each i from 2 to limit:
    #   if lpf[i] == 0, i is prime, set lpf[i] = i
    #   if i*i <= limit, mark multiples of i starting from i*i where lpf is still 0
    
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            lpf[i] = i
            if i * i <= limit:
                start = i * i
                # Vectorized: set lpf[j] = i where lpf[j] == 0
                mask = lpf[start:limit + 1:i] == 0
                if np.any(mask):
                    lpf[start:limit + 1:i][mask] = i
    return lpf


# Determine the range needed
min_val = min(lo for lo, hi in P['gaps'])
max_val = max(hi for lo, hi in P['gaps'])

# Precompute LPF for the entire range of interest
_lpf = _compute_lpf(max_val)


def construct():
    global _start_time, _DEADLINE
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() - _start_time > _DEADLINE:
            break
        
        gap = hi - lo
        
        # Find all candidates m in (lo, hi) where lpf[m] >= gap
        # Select the one with the HIGHEST lpf[m] for maximum robustness
        best_candidate = None
        best_lpf = 0
        
        for m in range(lo + 1, hi):
            if time.time() - _start_time > _DEADLINE:
                break
            lpf_m = int(_lpf[m])
            if lpf_m >= gap:
                if lpf_m > best_lpf:
                    best_lpf = lpf_m
                    best_candidate = m
        
        if best_candidate is not None:
            result[i] = int(best_candidate)
    
    # Ensure all entries are Python int or None
    return [int(x) if x is not None else None for x in result]
