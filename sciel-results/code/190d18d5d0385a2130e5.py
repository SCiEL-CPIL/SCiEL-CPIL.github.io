"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def construct():
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    # Precompute least prime factor (LPF) table up to max value needed.
    # We need LPF for all m in (lo, hi) for each gap. Max hi is 10289.
    max_val = max(hi for _, hi in gaps)
    # Sieve for smallest prime factor
    lpf = list(range(max_val + 1))
    for i in range(2, isqrt(max_val) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if lpf[j] == j:
                    lpf[j] = i

    for i, (lo, hi) in enumerate(gaps):
        gap_size = hi - lo
        # Find any m in (lo, hi) with lpf[m] >= gap_size
        found = False
        for m in range(lo + 1, hi):
            if lpf[m] >= gap_size:
                result[i] = int(m)
                found = True
                break
        if not found:
            result[i] = None

    return result


# EVOLVE-BLOCK-END
