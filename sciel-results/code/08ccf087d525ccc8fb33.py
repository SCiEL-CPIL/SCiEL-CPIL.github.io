import time
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    MAX_N = 10000000
    
    # Use Python list for sigma - faster than numpy for this pattern
    sigma = [0] * (MAX_N + 1)
    
    # Sieve for sigma
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1, i):
            sigma[j] += i
        if time.time() > deadline - 15:
            break
    
    if time.time() > deadline - 10:
        return [11, 6]
    
    # Group by sigma
    fibers = defaultdict(list)
    for n in range(1, MAX_N + 1):
        fibers[sigma[n]].append(n)
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    best_a, best_b = 11, 6
    best_err = abs(11/6 - 1.5)
    
    for s_val, nums in fibers.items():
        if time.time() > deadline - 2:
            break
        if len(nums) < 2:
            continue
        nums.sort()
        for b in nums:
            if time.time() > deadline - 1:
                break
            target = 1.5 * b
            pos = bisect.bisect_left(nums, target)
            for idx in (pos-1, pos):
                if 0 <= idx < len(nums):
                    a = nums[idx]
                    if a != b:
                        err = abs(a/b - 1.5)
                        if err < best_err:
                            best_err = err
                            best_a, best_b = a, b
                            if err == 0.0:
                                return [int(best_a), int(best_b)]
    
    return [int(best_a), int(best_b)]
