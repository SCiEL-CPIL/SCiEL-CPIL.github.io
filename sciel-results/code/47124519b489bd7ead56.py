"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Find the range we need to sieve
    all_lo = min(lo for lo, hi in P['gaps'])
    all_hi = max(hi for lo, hi in P['gaps'])
    
    # Sieve for least prime factor up to all_hi using NumPy for speed
    size = all_hi + 1
    lpf = np.arange(size, dtype=np.int32)
    
    # Sieve of Eratosthenes to fill lpf
    limit = isqrt(all_hi)
    for i in range(2, limit + 1):
        if lpf[i] == i:  # i is prime
            # Mark multiples starting from i*i
            start = i * i
            lpf[start::i] = np.where(lpf[start::i] == np.arange(start, size, i), i, lpf[start::i])
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        
        # Search for m in (lo, hi) such that lpf[m] >= gap
        # Convert to Python list for the small range
        candidates = range(lo + 1, hi)
        for m in candidates:
            if lpf[m] >= gap:
                result[i] = int(m)
                break
    
    # Ensure all values are Python int or None
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))
            
    return final_result
# EVOLVE-BLOCK-END
