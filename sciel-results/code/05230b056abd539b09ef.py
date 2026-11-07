import numpy as np
import time
from collections import defaultdict
import bisect

P = {'maximum': 10000000, 'target': [3, 2]}

def construct():
    N = P['maximum']
    target_ratio = 1.5
    
    # Deadline guard
    start_time = time.time()
    timeout = 85.0  # 5s margin
    
    # 1. Compute sigma(n) for n in [1, N]
    # Using NumPy for the inner loop
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve
    # This loop runs N times.
    # To speed up, we can process in chunks or just rely on NumPy speed.
    # 10^7 iterations of a simple NumPy op might be ~5-10s.
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() - start_time > timeout:
        # Fallback to a simple valid pair if sieve is too slow (unlikely)
        return [11, 6]

    # 2. Group indices by sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        s = sigma[n]
        groups[s].append(n)
        
    if time.time() - start_time > timeout:
        return [11, 6]

    # 3. Find best pair
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Iterate through groups
    # We only care about groups with at least 2 elements
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
            
        # Check time
        if time.time() - start_time > timeout:
            break
            
        # Sort the numbers in this group
        # They are already added in increasing order of n, so they are sorted?
        # Yes, we iterated n from 1 to N, so the lists are naturally sorted.
        # So no need to sort.
        
        # For each b in nums, find a in nums closest to 1.5 * b
        # Since nums is sorted, we can use bisect.
        for b in nums:
            target_a = 1.5 * b
            
            # Find insertion point
            # bisect_left returns the index where target_a would be inserted to maintain order
            idx = bisect.bisect_left(nums, target_a)
            
            # Check idx and idx - 1
            candidates = []
            if idx < len(nums):
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx - 1])
                
            # Also check idx+1? No, closest is either left or right of insertion point.
            # Wait, if target is 10.5, and list is [10, 11], idx for 10.5 is 1 (insert before 11).
            # Candidates: 11 (idx 1) and 10 (idx 0). Correct.
            
            for a in candidates:
                if a == b:
                    continue
                # Calculate error |a/b - 1.5|
                # Using float might be enough, but task says "Factorization and rational error are exact"
                # "score is the binary64 serialization of that rational"
                # So we compute the float error.
                err = abs(a / b - 1.5)
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
                    
    # Ensure return types are Python int
    return [int(best_a), int(best_b)]
