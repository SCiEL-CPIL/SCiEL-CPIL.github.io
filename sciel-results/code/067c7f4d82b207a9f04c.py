import time
import numpy as np
from collections import defaultdict

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

    # Group numbers by sigma value
    # Using a dict of lists is memory heavy but fast enough for 10^7
    groups = defaultdict(list)
    for n in range(1, N + 1):
        s = int(sigma[n])
        groups[s].append(n)
        
    if time.time() > deadline - 5:
        return [11, 6]

    best_err = float('inf')
    best_pair = None
    best_num = float('inf') # Numerator |2a-3b|
    best_den = 0           # Denominator 2b
    
    # To optimize, we can sort groups by max value descending?
    # Or just iterate.
    
    # Pre-sorting the keys by max value might help pruning?
    # Let's just iterate and prune based on current best.
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
            
        if time.time() > deadline - 2:
            break
            
        # The list 'nums' is not sorted. We need it sorted for binary search.
        nums.sort()
        
        # Pruning: 
        # Max possible b in this group is nums[-1].
        # Min possible error in this group is 1 / (2 * nums[-1]) (if numerator 1 is possible)
        # If 1 / (2 * nums[-1]) >= best_err, we can skip this group?
        # Not exactly, because numerator could be 1, 2, etc.
        # But if 1 / (2 * nums[-1]) > best_err, then even the best case (num=1) is worse.
        # So if best_err < 1.0 / (2.0 * nums[-1]), skip.
        
        if best_err < 1.0 / (2.0 * nums[-1]):
            continue
            
        n_len = len(nums)
        
        # Two pointer or binary search
        # For each b in nums, find a in nums closest to 1.5 * b
        # Since nums is sorted, we can use bisect.
        
        for b in nums:
            target = 1.5 * b
            # Find insertion point
            # bisect_left(nums, target)
            import bisect
            idx = bisect.bisect_left(nums, target)
            
            # Check idx and idx-1
            candidates = []
            if idx < n_len:
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx-1])
                
            for a in candidates:
                if a == b:
                    continue # a and b must be distinct? Task says "a, b positive integers", doesn't explicitly say distinct, but sigma(a)=sigma(b) and a!=b is implied by "ratio" context usually, and if a=b, ratio is 1, error 0.5. But if a=b, |a/b - 1.5| = 0.5. If we can find closer, it's better. Also, if a=b, 2a-3b = -b, num=b, den=2b, err=0.5.
                    # Actually, if a=b, sigma(a)=sigma(b) is trivially true. 
                    # But we want to minimize |a/b - 1.5|. 
                    # If a=b, error is 0.5.
                    # We can probably find better.
                    pass
                
                num = abs(2*a - 3*b)
                if num == 0:
                    # Invalid per task: "A raw value of exactly 0 cannot be scored and counts as invalid."
                    # So we must skip a/b = 1.5 exactly.
                    continue
                    
                den = 2*b
                err = num / den
                
                if err < best_err:
                    best_err = err
                    best_pair = (a, b)
                    best_num = num
                    best_den = den

    if best_pair is not None:
        return [int(best_pair[0]), int(best_pair[1])]
    else:
        return [11, 6]
