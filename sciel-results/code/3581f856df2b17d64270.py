import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    LO = 10000
    HI = 1000000
    MAX_VAL = HI + 1  # 1000001, since we need lpf up to n+1 where n can be 1000000

    # Sieve for largest prime factor using NumPy
    # lpf[i] will hold the largest prime factor of i
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)

    # For each number p from 2 to MAX_VAL, if lpf[p] == 0, p is prime
    # Then mark all multiples of p with lpf = p
    # Since we iterate p in increasing order, the last assignment gives the largest prime factor

    # Optimize: iterate only up to MAX_VAL, but use NumPy slicing for marking
    # Add deadline check to satisfy explicit guard requirement
    for p in range(2, MAX_VAL + 1):
        if time.time() > deadline:
            break
        if lpf[p] == 0:  # p is prime
            lpf[p::p] = p

    # Compute max(lpf[n], lpf[n+1]) for all n in [LO, HI]
    # n ranges from LO to HI, so n+1 ranges from LO+1 to HI+1
    n_arr = np.arange(LO, HI + 1)
    vals = np.maximum(lpf[LO:HI + 1], lpf[LO + 1:HI + 2])

    # Find the index of the minimum value
    min_idx = np.argmin(vals)
    best_n = int(n_arr[min_idx])
    best_val = int(vals[min_idx])

    return best_n
