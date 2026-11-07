"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Determine the maximum value we need to check
    max_val = max(hi for lo, hi in P['gaps'])
    
    # Sieve for least prime factor (lpf) up to max_val
    # lpf[i] = least prime factor of i, or i if i is prime
    lpf = list(range(max_val + 1))
    for i in range(2, isqrt(max_val) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # We need an integer m strictly between lo and hi such that lpf[m] >= gap
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                break
        # If deadline is approaching, we could try to be smarter about remaining gaps
        if time.time() > deadline:
            break
    
    # Ensure all entries are Python int or None (no numpy types or bools)
    for i in range(len(result)):
        if result[i] is not None:
            result[i] = int(result[i])
    
    return result
# EVOLVE-BLOCK-END
