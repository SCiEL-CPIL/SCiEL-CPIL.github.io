"""Approximate a ratio inside a divisor-sum fiber."""
import time
import numpy as np
from bisect import bisect_left

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85
    N = 10_000_000
    
    # Compute sigma(n) for n in [1, N] using numpy sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        if time.time() > deadline - 35:
            break
        sigma[i::i] += i
    
    if time.time() > deadline - 30:
        return [11, 6]
    
    # Group indices by sigma value
    groups = {}
    sigma_list = sigma.tolist()
    for n in range(1, N + 1):
        s = sigma_list[n]
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]
    
    if time.time() > deadline - 10:
        return [11, 6]
    
    # Search for best pair (a, b) with sigma(a) == sigma(b) minimizing |a/b - 3/2|
    # error = |2a - 3b| / (2b)
    best_num = 1
    best_den = 3  # seed: |11/6 - 3/2| = 1/3
    best_a, best_b = 11, 6
    
    for s, lst in groups.items():
        if time.time() > deadline - 3:
            break
        K = len(lst)
        if K < 2:
            continue
        for idx in range(K):
            b = lst[idx]
            target = int(1.5 * b)
            pos = bisect_left(lst, target)
            for p in (pos - 1, pos, pos + 1):
                if 0 <= p < K:
                    a = lst[p]
                    if a == b:
                        continue
                    num = abs(2 * a - 3 * b)
                    if num == 0:
                        continue
                    den = 2 * b
                    if num * best_den < best_num * den:
                        best_num = num
                        best_den = den
                        best_a, best_b = a, b
    
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
