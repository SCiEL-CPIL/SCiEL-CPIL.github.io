import math
import time
import numpy as np

def construct():
    target = 123457
    start_time = time.time()
    deadline = 85.0  # 5s safety margin before 90s kill
    
    # Precompute largest prime factor for all numbers up to target using NumPy
    lpf = np.zeros(target + 1, dtype=np.int64)
    lpf[1] = 1
    
    # Sieve for largest prime factor using vectorized operations
    for i in range(2, target + 1):
        if time.time() > deadline:
            break
        if lpf[i] == 0:
            # i is prime - vectorize the multiple assignment
            multiples = np.arange(i, target + 1, i)
            lpf[multiples] = i
    
    best_max_p = float('inf')
    best_a = 1
    best_b = target - 1
    
    # Since a+b is odd, one is even, one is odd.
    # The even number has factor 2.
    # The odd number (if >1) has odd prime factors.
    # We want to minimize the max prime factor.
    
    # Iterate a from 1 to target-1
    # We can stop at target // 2 + 1 because (a, b) and (b, a) yield the same max(P(a), P(b))
    half = target // 2 + 1
    lpf_list = lpf.tolist()  # Convert to list for faster scalar access in Python loop
    
    for a in range(1, half):
        if time.time() > deadline:
            break
        b = target - a
        pa = lpf_list[a]
        pb = lpf_list[b]
        cur_max = pa if pa > pb else pb
        if cur_max < best_max_p:
            best_max_p = cur_max
            best_a = a
            best_b = b
            # Theoretical min for max(P(a),P(b)) given a+b is odd:
            # Case 2: a>1, b>1. One is even (P>=2), one is odd (P>=3). So min max is 3.
            if best_max_p == 3:
                break
                
    return [best_a, best_b]
