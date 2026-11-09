import time
import numpy as np
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 10_000_000
    
    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline - 10:
        return [11, 6]

    # Group numbers by sigma value using a plain dict to reduce overhead
    groups = {}
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]
        
    if time.time() > deadline - 5:
        return [11, 6]

    best_err = float('inf')
    best_pair = None
    
    # Sort groups by max value descending for better pruning
    group_items = [(s, nums) for s, nums in groups.items() if len(nums) >= 2]
    group_items.sort(key=lambda x: max(x[1]), reverse=True)
    
    for s, nums in group_items:
        if time.time() > deadline - 2:
            break
            
        # The list 'nums' is not sorted. We need it sorted for binary search.
        nums.sort()
        
        # Pruning: 
        # Max possible b in this group is nums[-1].
        # Min possible error in this group is 1 / (2 * nums[-1]) (if numerator 1 is possible)
        # If best_err < 1.0 / (2.0 * nums[-1]), skip.
        if best_err < 1.0 / (2.0 * nums[-1]):
            continue
            
        n_len = len(nums)
        
        # For each b in nums, find a in nums closest to 1.5 * b
        for b in nums:
            target = 1.5 * b
            idx = bisect.bisect_left(nums, target)
            
            # Check idx and idx-1
            if idx < n_len:
                a = nums[idx]
                if a != b:
                    num = abs(2*a - 3*b)
                    if num != 0:
                        err = num / (2*b)
                        if err < best_err:
                            best_err = err
                            best_pair = (a, b)
            if idx > 0:
                a = nums[idx-1]
                if a != b:
                    num = abs(2*a - 3*b)
                    if num != 0:
                        err = num / (2*b)
                        if err < best_err:
                            best_err = err
                            best_pair = (a, b)

    if best_pair is not None:
        return [int(best_pair[0]), int(best_pair[1])]
    else:
        return [11, 6]
