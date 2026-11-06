"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s to be safe under 90s kill
    target = 123457
    max_n = target - 1  # 123456
    
    # Efficient sieve to compute largest prime factor using only primes up to sqrt(max_n)
    # Then for each number, divide out small primes and track the largest factor
    lpf = np.zeros(max_n + 1, dtype=np.int32)
    lpf[1] = 1
    
    # Sieve of Eratosthenes to find all primes up to max_n
    # For each prime p, mark multiples: lpf[multiple] = max(lpf[multiple], p)
    # Since we iterate primes in increasing order, later primes overwrite earlier ones
    # So the final lpf value is the largest prime factor
    
    # Use a bytearray for is_prime for speed
    is_prime = np.ones(max_n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    # Find primes up to sqrt(max_n) first, then handle larger primes
    # Actually, the simplest correct approach: iterate i from 2 to max_n,
    # if lpf[i] == 0 then i is prime, mark multiples with i
    
    # Optimize: only iterate i up to max_n, but skip composites
    # The key optimization: for prime i, only update multiples that don't already
    # have a larger factor. Since we go in increasing order, i is always <= current max factor
    # for any multiple that already has a factor. So we only need to set lpf[j] = i
    # when lpf[j] == 0 (i is the first/only prime factor found so far, meaning it's the largest
    # because no larger prime divides j yet... wait, that's wrong).
    
    # Correct approach: iterate i from 2 to max_n. If i is prime (lpf[i]==0),
    # then for each multiple j = i, 2i, 3i, ..., set lpf[j] = i.
    # Since we iterate in increasing order, the LAST prime that divides j will be the largest.
    # So we just overwrite: lpf[j] = i for each prime i dividing j.
    # Final lpf[j] = largest prime factor of j.
    
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            # Mark all multiples of i with i as a prime factor
            # Since we go in increasing order, this overwrites with larger primes
            for j in range(i, max_n + 1, i):
                lpf[j] = i
    
    # Check time - if sieve took too long, we still have a valid result
    if time.time() > deadline - 5:
        # Fallback: just return a reasonable split
        return [target // 2, target - target // 2]
    
    # Vectorized search: compute max(lpf[a], lpf[target-a]) for all a in [1, target-1]
    a_arr = np.arange(1, target)
    b_arr = target - a_arr
    pa = lpf[a_arr]
    pb = lpf[b_arr]
    cur_max = np.maximum(pa, pb)
    
    best_idx = np.argmin(cur_max)
    best_max = int(cur_max[best_idx])
    best_a = int(a_arr[best_idx])
    best_b = int(b_arr[best_idx])
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
