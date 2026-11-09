"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _lpf_sieve(n):
    """Return list lpf where lpf[i] = least prime factor of i, for i in 0..n."""
    lpf = list(range(n + 1))
    r = isqrt(n)
    for i in range(2, r + 1):
        if lpf[i] == i:  # i is prime
            ii = i * i
            for j in range(ii, n + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    return lpf

def construct():
    deadline = time.time() + 85.0
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    max_val = max(hi for _, hi in gaps)
    lpf = _lpf_sieve(max_val)

    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        g = hi - lo
        for m in range(lo + 1, hi):
            if lpf[m] >= g:
                result[i] = int(m)
                break

    return result

# EVOLVE-BLOCK-END
