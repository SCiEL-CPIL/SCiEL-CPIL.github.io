"""Program to find a,b with sigma(a)=sigma(b) minimizing |a/b - 3/2|, where error must be > 0."""
import math
import time

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

MAX_N = 10_000_000
TIME_LIMIT = 75.0  # safety margin (evaluator kills at 90s)

def construct():
    t_start = time.time()
    deadline = t_start + TIME_LIMIT

    # --- Build sigma (divisor sum) via vectorized sieve up to MAX_N ---
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d

    # Convert to list for faster scalar access in the search loop
    sigma_list = sigma.tolist()

    best_err = float('inf')
    best_pair = (1, 1)

    # --- Phase 1: Sweep all b, checking a candidates in a band around 3b/2 ---
    # For each b, target_a = 3b/2. We want integer a near target_a with sigma[a]==sigma[b].
    # Check a window of radius W around target_a.
    W = 3  # search radius in each direction from floor/ceil of 3b/2

    for b in range(1, MAX_N + 1):
        if time.time() > deadline:
            break
        sb = sigma_list[b]
        target = 3 * b / 2.0
        a_center = int(round(target))
        
        for offset in range(-W, W + 1):
            a = a_center + offset
            if 1 <= a <= MAX_N and a != b:
                if sigma_list[a] == sb:
                    err = abs(a / b - 1.5)
                    if 0 < err < best_err:
                        best_err = err
                        best_pair = (a, b)

    # --- Phase 2: Random sampling with wider band ---
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        random.seed(42)
        W2 = 20  # wider search radius for random phase
        for _ in range(min(2000000, int(remaining_time * 50000))):
            if time.time() > deadline - 1.0:
                break
            b = random.randint(1, MAX_N)
            sb = sigma_list[b]
            target = 3 * b / 2.0
            a_center = int(round(target))
            
            for offset in range(-W2, W2 + 1):
                a = a_center + offset
                if 1 <= a <= MAX_N and a != b:
                    if sigma_list[a] == sb:
                        err = abs(a / b - 1.5)
                        if 0 < err < best_err:
                            best_err = err
                            best_pair = (a, b)

    # --- Phase 3: Targeted search for large b values where relative error can be tiny ---
    # For large b, even a small absolute difference |2a-3b| gives tiny relative error.
    # Focus on b > 1000000 with wider window.
    remaining_time = deadline - time.time()
    if remaining_time > 3:
        W3 = 50
        for b in range(max(1000001, MAX_N - 2000000), MAX_N + 1):
            if time.time() > deadline - 1.0:
                break
            sb = sigma_list[b]
            target = 3 * b / 2.0
            a_center = int(round(target))
            
            for offset in range(-W3, W3 + 1):
                a = a_center + offset
                if 1 <= a <= MAX_N and a != b:
                    if sigma_list[a] == sb:
                        err = abs(a / b - 1.5)
                        if 0 < err < best_err:
                            best_err = err
                            best_pair = (a, b)

    a, b = best_pair
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
