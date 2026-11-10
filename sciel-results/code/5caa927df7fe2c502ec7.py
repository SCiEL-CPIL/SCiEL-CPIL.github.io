"""Refined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import bisect
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    MAX_N = 10000000

    # Step 1: Build spf
    spf = np.zeros(MAX_N + 1, dtype=np.int32)
    for i in range(2, MAX_N + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= MAX_N:
                for j in range(i * i, MAX_N + 1, i):
                    if spf[j] == 0:
                        spf[j] = i

    # Step 2: Compute sigma(n)
    p_pow = np.ones(MAX_N + 1, dtype=np.int64)
    rem = np.arange(MAX_N + 1, dtype=np.int64)
    sum_power = np.ones(MAX_N + 1, dtype=np.int64)
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    sigma[1] = 1
    for n in range(2, MAX_N + 1):
        p = spf[n]
        if n % (p * p) == 0:
            p_pow[n] = p_pow[n // p] * p
            rem[n] = rem[n // p]
            sum_power[n] = sum_power[n // p] + p_pow[n]
        else:
            p_pow[n] = p
            rem[n] = n // p
            sum_power[n] = 1 + p
        sigma[n] = sigma[rem[n]] * sum_power[n]

    # Step 3: Group by sigma
    fibers = defaultdict(list)
    for n in range(1, MAX_N + 1):
        fibers[sigma[n]].append(n)

    # Step 4: Find best pair
    best_a = 11
    best_b = 6
    best_err = abs(11/6 - 3/2)

    for s_val, nums in fibers.items():
        if time.time() > deadline - 5:
            break
        if len(nums) < 2:
            continue
        max_b = max(nums)
        if max_b <= 1 / (2 * best_err):
            continue
        nums.sort()
        for b in reversed(nums):
            target_a = 1.5 * b
            pos = bisect.bisect_left(nums, target_a)
            candidates = []
            if pos > 0:
                candidates.append(nums[pos-1])
            if pos < len(nums):
                candidates.append(nums[pos])
            for a in candidates:
                err = abs(a / b - 3/2)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    if err == 0.0:
                        return [int(best_a), int(best_b)]

    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
