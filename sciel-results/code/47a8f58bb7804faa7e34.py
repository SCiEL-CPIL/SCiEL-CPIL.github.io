import time
import numpy as np

_start_time = time.time()
_DEADLINE = 85.0

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _compute_lpf(limit):
    """Compute least prime factor for all numbers up to limit using vectorized sieve."""
    lpf = np.zeros(limit + 1, dtype=np.int64)
    lpf[1] = 1
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            lpf[i] = i
            if i * i <= limit:
                # Vectorized marking: for each prime i, mark all multiples starting from i*i
                start = i * i
                lpf[start:limit + 1:i] = np.where(lpf[start:limit + 1:i] == 0, i, lpf[start:limit + 1:i])
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
        found = None
        
        # Iterate through candidates in (lo, hi)
        # Using the precomputed LPF array for O(1) check
        for m in range(lo + 1, hi):
            if _lpf[m] >= gap:
                found = m
                break
        
        if found is not None:
            result[i] = int(found)
            
    return [int(x) if x is not None else None for x in result]
