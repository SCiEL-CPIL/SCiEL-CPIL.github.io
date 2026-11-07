# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _build_small_primes(limit):
    """Build list of primes up to limit using sieve."""
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin
    
    # Precompute small primes up to 200
    small_primes = _build_small_primes(200)
    small_primes_arr = np.array(small_primes, dtype=np.int64)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        
        # Generate all candidates in (lo, hi)
        candidates = np.arange(lo + 1, hi, dtype=np.int64)
        if len(candidates) == 0:
            continue
        
        # For each candidate, find if LPF >= gap
        # LPF(m) >= gap means m is not divisible by any prime < gap
        primes_below_gap = small_primes_arr[small_primes_arr < gap]
        
        if len(primes_below_gap) == 0:
            result[i] = int(candidates[0])
            continue
        
        # Check which candidates are NOT divisible by any prime < gap
        valid_mask = np.ones(len(candidates), dtype=bool)
        
        for p in primes_below_gap:
            if p * p > candidates[-1]:
                break
            valid_mask &= (candidates % p != 0)
            if not np.any(valid_mask):
                break
        
        if np.any(valid_mask):
            first_valid_idx = np.argmax(valid_mask)
            result[i] = int(candidates[first_valid_idx])
    
    return result
# EVOLVE-BLOCK-END
