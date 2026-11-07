"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_lpf(n):
    """Return an array lpf where lpf[i] is the least prime factor of i (or i if prime)."""
    if n < 2:
        return list(range(n + 1))
    
    # Initialize lpf with the value itself
    lpf = list(range(n + 1))
    
    # Sieve of Eratosthenes variant to find least prime factors
    limit = isqrt(n)
    for i in range(2, limit + 1):
        if lpf[i] == i:  # i is prime
            # Mark multiples of i starting from i*i
            # If lpf[j] is still j, it means no smaller prime factor was found
            for j in range(i * i, n + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
                    
    return lpf

def construct():
    start_time = time.time()
    # Safety margin: 85 seconds out of 90
    deadline = start_time + 85

    n_gaps = len(P['gaps'])
    result = [None] * n_gaps

    # Determine the maximum value needed for the sieve
    max_val = max(hi for lo, hi in P['gaps'])
    
    # Precompute least prime factors for all numbers up to max_val
    # This allows O(1) lookup for lpf(m)
    lpf = _sieve_lpf(max_val)

    for i in range(n_gaps):
        lo, hi = P['gaps'][i]
        gap = hi - lo
        
        # Check time budget
        if time.time() > deadline:
            break
            
        # Find the first integer m in (lo, hi) such that lpf[m] >= gap
        # m must be strictly between lo and hi
        found_witness = False
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = int(m)
                found_witness = True
                break
                
        # If no witness found, result[i] remains None

    return result

# EVOLVE-BLOCK-END
