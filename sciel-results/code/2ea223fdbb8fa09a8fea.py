"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Leave 15s safety margin before 90s kill

    # Precompute smallest prime factor for numbers up to max candidate + 1
    max_candidate = max(hi for lo, hi in P['gaps'])
    spf = list(range(max_candidate + 1))
    for i in range(2, isqrt(max_candidate) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_candidate + 1, i):
                if spf[j] == j:
                    spf[j] = i

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # We need m in (lo, hi) such that least_prime_factor(m) >= gap
        # Since the gap is small (max 20), linear scan is efficient and exhaustive
        # Direct SPF lookup is O(1) per candidate, so this is very fast
        for m in range(lo + 1, hi):
            if spf[m] >= gap:
                result[i] = int(m)
                break

    # Ensure all entries are Python int or None
    return [int(x) if x is not None else None for x in result]
# EVOLVE-BLOCK-END
