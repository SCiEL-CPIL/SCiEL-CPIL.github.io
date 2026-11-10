"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    MAX_N = 10000000

    # Sieve for sigma(n) = sum of divisors of n
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    for i in range(1, MAX_N + 1):
        sigma[i::i] += i

    if time.time() > deadline - 5.0:
        return [11, 6]

    # Group numbers by their sigma value
    sigma_list = sigma.tolist()
    sigma_to_numbers = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_to_numbers[sigma_list[n]].append(n)

    if time.time() > deadline - 3.0:
        return [11, 6]

    best_err = abs(11 / 6 - 3 / 2)
    best_a, best_b = 11, 6
    target = 1.5

    for s, nums in sigma_to_numbers.items():
        if len(nums) < 2:
            continue
        nums_sorted = sorted(nums)
        for b in nums_sorted:
            ideal_a = 1.5 * b
            idx = bisect.bisect_left(nums_sorted, ideal_a)
            candidates = []
            if idx < len(nums_sorted):
                candidates.append(nums_sorted[idx])
            if idx > 0:
                candidates.append(nums_sorted[idx - 1])
            for a in candidates:
                if a == b:
                    continue
                err = abs(a / b - target)
                if err < best_err:
                    best_err = err
                    best_a, best_b = a, b

    return [int(best_a), int(best_b)]
# EVOLVE-BLOCK-END
