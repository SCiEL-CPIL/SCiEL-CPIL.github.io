import time
import numpy as np

_start_time = time.time()
_DEADLINE = 85.0

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes(n):
    """Return list of primes up to n (inclusive)."""
    if n < 2:
        return []
    is_p = bytearray([1]) * (n + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, n + 1) if is_p[i]]

# Precompute primes up to the maximum gap
_max_gap = max(hi - lo for lo, hi in P['gaps'])
_primes = _sieve_primes(_max_gap)

def _find_witness(lo, hi):
    """Find an integer m in (lo, hi) whose least prime factor is >= gap = hi - lo.
    
    Uses an interval sieve: for each prime p < gap, mark all multiples of p
    in the open interval. Survivors have no prime factor < gap, so their least
    prime factor is >= gap (or they are prime themselves).
    """
    gap = hi - lo
    length = gap - 1  # number of integers in (lo, hi)
    if length <= 0:
        return None
    
    # marked[i] = True if (lo+1+i) is divisible by some prime < gap
    marked = np.zeros(length, dtype=bool)
    
    for p in _primes:
        if p >= gap:
            break
        # Find the smallest multiple of p in [lo+1, hi-1]
        start = lo + 1
        first = ((start + p - 1) // p) * p
        if first >= hi:
            continue
        # Mark all multiples of p in [first, hi) using vectorized indexing
        indices = np.arange(first - (lo + 1), length, p)
        marked[indices] = True
    
    # Collect survivors
    survivor_indices = np.where(~marked)[0]
    
    if len(survivor_indices) == 0:
        return None
    
    # Pick the middle survivor for robustness
    mid = len(survivor_indices) // 2
    return int(lo + 1 + survivor_indices[mid])

def construct():
    global _start_time, _DEADLINE
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() - _start_time > _DEADLINE:
            break
        w = _find_witness(lo, hi)
        if w is not None:
            result[i] = w
    
    # Ensure all entries are Python int or None
    return [int(x) if x is not None else None for x in result]
