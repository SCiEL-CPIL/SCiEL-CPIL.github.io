import time
import sys
import numpy as np
from bisect import bisect_left

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n <= N using a sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for i in range(1, N + 1):
        sigma[i::i] += i
        if i % 1000000 == 0 and time.time() > deadline - 15:
            return [11, 6]
    
    if time.time() > deadline - 10:
        return [11, 6]
    
    # 2. Group numbers by sigma value
    groups = {}
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    # 3. Find best pair minimizing |a/b - 3/2|
    best_diff = float('inf')
    best_a = 11
    best_b = 6
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
            
        nums.sort()
        m = len(nums)
        
        for j in range(m):
            b = nums[j]
            target = 1.5 * b
            pos = bisect_left(nums, target)
            
            if pos < m:
                a = nums[pos]
                val = abs(2*a - 3*b) / (2*b)
                if val < best_diff:
                    best_diff = val
                    best_a = a
                    best_b = b
            if pos > 0:
                a = nums[pos-1]
                val = abs(2*a - 3*b) / (2*b)
                if val < best_diff:
                    best_diff = val
                    best_a = a
                    best_b = b
        
        if time.time() > deadline - 2:
            break
    
    return [int(best_a), int(best_b)]
# EVOLVE-BLOCK-END
