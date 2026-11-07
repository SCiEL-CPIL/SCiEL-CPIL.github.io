# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _lpf_sieve(n):
    """Least prime factor array up to n (1-indexed). lpf[x] = smallest prime dividing x, or x if prime."""
    lpf = list(range(n + 1))
    if n >= 1:
        lpf[1] = 1
    for i in range(2, isqrt(n) + 1):
        if lpf[i] == i:  # i is prime
            step = i
            start = i * i
            for j in range(start, n + 1, step):
                if lpf[j] == j:
                    lpf[j] = i
    return lpf


def construct():
    deadline = time.time() + 45.0  # safety margin under 60s
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    max_val = max(hi for lo, hi in gaps)
    lpf = _lpf_sieve(max_val)

    # For each pair, find a candidate m in (lo, hi) with lpf[m] >= gap
    # Try multiple candidates per gap to be robust; pick the first valid one.
    for i, (lo, hi) in enumerate(gaps):
        gap = hi - lo
        if time.time() > deadline:
            break
        # Search candidates; prefer middle then outward to vary witnesses
        candidates = range(lo + 1, hi)
        for m in candidates:
            if lpf[m] >= gap:
                result[i] = int(m)
                break
    return result
# EVOLVE-BLOCK-END
