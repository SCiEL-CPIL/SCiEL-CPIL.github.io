import time
import numpy as np
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s
    
    # We'll try progressively larger ranges
    # Start with a moderate range that's computable quickly
    N = 2_000_000  # Start with 2 million
    
    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        # Check time periodically
        if d % 200_000 == 0:
            if time.time() > deadline - 5:
                break
    
    # Build groups: dict from sigma value -> list of n
    from collections import defaultdict
    groups = defaultdict(list)
    for n in range(1, N + 1):
        s = int(sigma[n])
        groups[s].append(n)
    
    # For each group, find pairs with ratio closest to 3/2
    best_error = float('inf')
    best_pair = (11, 6)  # default fallback
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        nums.sort()
        for i, b in enumerate(nums):
            target = 3 * b / 2
            # Binary search for closest to target
            pos = bisect_left(nums, target)
            for j in [pos - 1, pos]:
                if 0 <= j < len(nums) and j != i:
                    a = nums[j]
                    if a == b:
                        continue
                    ratio = a / b
                    err = abs(ratio - 1.5)
                    if err < best_error:
                        best_error = err
                        best_pair = (a, b)
        
        if time.time() > deadline - 3:
            break
    
    a, b = best_pair
    # Ensure positive integers <= 10^7
    a = int(a)
    b = int(b)
    if a <= 0 or b <= 0 or a > 10_000_000 or b > 10_000_000:
        return [11, 6]
    return [a, b]
