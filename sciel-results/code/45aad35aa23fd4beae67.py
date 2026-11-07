"""Model-and-optimize rewrite for the consecutive-prime-gap witness task."""
import time
import math
import numpy as np

_START_TIME = time.time()
_DEADLINE = 80.0  # explicit safety margin before the 90s kill

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _build_lpf(n):
    """Build least prime factor sieve up to n. lpf[i] = smallest prime factor of i."""
    lpf = list(range(n + 1))
    if n >= 0:
        lpf[0] = 0
    if n >= 1:
        lpf[1] = 1
    for i in range(2, int(math.isqrt(n)) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    return lpf


# Precompute LPF up to the maximum prime in our range
_max_val = max(hi for lo, hi in P['gaps'])
_lpf = _build_lpf(_max_val)


def construct():
    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() - _START_TIME > _DEADLINE:
            break
        gap = hi - lo
        # Find first m in (lo, hi) with LPF(m) >= gap
        found = None
        for m in range(lo + 1, hi):
            if _lpf[m] >= gap:
                found = m
                break
        if found is not None:
            result[i] = int(found)
    return result
