import time
from collections import defaultdict
import bisect

def construct():
    N = 10000000
    start = time.time()
    deadline = start + 80
    
    # Compute sigma(n) for all n up to N
    sigma = [0] * (N + 1)
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            sigma[m] += d
        if time.time() > deadline - 5:
            break
    
    # Group by sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        if time.time() > deadline - 3:
            break
        groups[sigma[n]].append(n)
    
    # Find best pair
    best_err = float('inf')
    best_pair = (11, 6)  # fallback from seed
    
    for s, nums in groups.items():
        if time.time() > deadline - 2:
            break
        if len(nums) < 2:
            continue
        nums.sort()
        for b in nums:
            if time.time() > deadline - 1:
                break
            target = 1.5 * b
            idx = bisect.bisect_left(nums, target)
            candidates = []
            if idx > 0:
                candidates.append(nums[idx-1])
            if idx < len(nums):
                candidates.append(nums[idx])
            for a in candidates:
                if a == b:
                    continue
                err = abs(a / b - 1.5)
                if err < best_err:
                    best_err = err
                    best_pair = (a, b)
                    if best_err == 0:
                        return [a, b]
    
    return list(best_pair)
