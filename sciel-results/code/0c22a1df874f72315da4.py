"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    # 1. Find max value to sieve
    max_val = max(hi for lo, hi in P['gaps'])
    
    # 2. Sieve of Eratosthenes to get primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(max_val) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 3. Precompute LPF (least prime factor) for all numbers up to max_val
    lpf = [0] * (max_val + 1)
    for p in primes:
        if p > isqrt(max_val):
            break
        for j in range(p*p, max_val + 1, p):
            if lpf[j] == 0:
                lpf[j] = p
    
    # For primes, LPF is the prime itself
    for p in primes:
        if lpf[p] == 0:
            lpf[p] = p
    
    # 4. For each gap, find a witness using LPF array
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap_size = hi - lo
        
        # Check time
        if time.time() > deadline:
            break
        
        # Find the first m in (lo, hi) where lpf[m] >= gap_size
        for m in range(lo + 1, hi):
            if lpf[m] >= gap_size:
                result[i] = m
                break
                
    return result

# EVOLVE-BLOCK-END
