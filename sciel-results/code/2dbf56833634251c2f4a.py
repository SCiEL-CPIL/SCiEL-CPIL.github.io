import sys
import time
import math
import numpy as np

MAX_N = 10000000

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # 1. Compute sigma(n) for all n <= MAX_N using numpy sieve
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)

    half = MAX_N // 2
    for i in range(1, half + 1):
        if time.time() > deadline - 25:
            # Fallback: use a precomputed pair
            return [11, 6]
        sigma[i::i] += i

    # For i > MAX_N/2, sigma[i::i] has only one element: sigma[i]
    if time.time() <= deadline - 25:
        sigma[half + 1:] += np.arange(half + 1, MAX_N + 1, dtype=np.int64)

    if time.time() > deadline - 10:
        return [11, 6]

    # 2. Targeted search: for each b, check a = (3b±k)/2 for small k
    best_a, best_b = 11, 6
    best_num, best_den = 4, 12  # |2*11 - 3*6| / (2*6) = 4/12 = 1/3

    max_k = 20  # search |2a-3b| <= max_k
    
    for b in range(MAX_N, 0, -1):
        if time.time() > deadline - 2:
            break
        
        # Early termination: if best error < 1/(2*b), no smaller b can improve
        if 2 * b * best_num < best_den:
            break
        
        sb = int(sigma[b])
        
        for k in range(1, max_k + 1):
            # a = (3b + k) / 2
            num1 = 3 * b + k
            if num1 % 2 == 0:
                a = num1 // 2
                if 1 <= a <= MAX_N and a != b:
                    if int(sigma[a]) == sb:
                        if k * best_den < best_num * (2 * b):
                            best_a, best_b = a, b
                            best_num, best_den = k, 2 * b
                            if best_num == 1:
                                # Can't do better than 1/(2b) for this b
                                # But a larger b might give 1/(2b') < 1/(2b)
                                # So continue searching
                                pass
            
            # a = (3b - k) / 2
            num2 = 3 * b - k
            if num2 % 2 == 0 and num2 > 0:
                a = num2 // 2
                if 1 <= a <= MAX_N and a != b:
                    if int(sigma[a]) == sb:
                        if k * best_den < best_num * (2 * b):
                            best_a, best_b = a, b
                            best_num, best_den = k, 2 * b
    
    return [int(best_a), int(best_b)]
