"""LPF-sieve based witness finder for consecutive prime gaps."""
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 82.0  # safety margin before 90s kill

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

# Precompute least prime factor (LPF) for all integers up to max prime in our range
_max_n = max(hi for lo, hi in P['gaps'])
_lpf = np.zeros(_max_n + 1, dtype=np.int64)

# Initialize LPF for all numbers as 0 (meaning not yet set, or prime)
# For sieve of LPF:
for i in range(2, _max_n + 1):
    if _lpf[i] == 0:  # i is prime
        _lpf[i] = i
        # Mark multiples
        for j in range(i * i, _max_n + 1, i):
            if _lpf[j] == 0:
                _lpf[j] = i

# For numbers that are still 0, they must be 1 or 0; handle edge cases
_lpf[0] = 0
_lpf[1] = 0


def _find_witness_via_lpf(lo, hi):
    """
    Find an integer m in (lo, hi) such that lpf(m) >= gap = hi - lo.
    Uses precomputed LPF array for O(1) lookup per candidate.
    Returns the middle valid candidate or None.
    """
    gap = hi - lo
    if gap <= 1:
        return None
    
    valid_indices = []
    for m in range(lo + 1, hi):
        # lpf(m) should be >= gap
        # If m is prime, lpf[m] = m, and since m < hi and m > lo, 
        # m >= gap only if gap is small. But actually we just check lpf[m] >= gap
        lpf_val = int(_lpf[m])
        if lpf_val >= gap:
            valid_indices.append(m)
    
    if not valid_indices:
        return None
    
    mid = len(valid_indices) // 2
    return int(valid_indices[mid])


def construct():
    global _start_time, _DEADLINE
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() - _start_time > _DEADLINE:
            break
        w = _find_witness_via_lpf(lo, hi)
        if w is not None:
            result[i] = int(w)
    
    return [int(x) if x is not None else None for x in result]
