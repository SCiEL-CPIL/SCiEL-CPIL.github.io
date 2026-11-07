import time
import math
import bisect
import numpy as np

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit (or 90s if specified, but 55s is safe)
    
    N = 10_000_000
    
    # 1. Compute sigma(n) using NumPy
    # sigma[n] = sum of divisors of n
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    # This is O(N log N) operations
    for i in range(1, N + 1):
        if time.time() - start_time > deadline - 5.0:
            # If sigma computation takes too long, return a fallback
            # Fallback: 11, 6 (sigma(11)=12, sigma(6)=12). Ratio 11/6 = 1.833. Error |1.833-1.5|=0.333. Score 3.
            # We can do better with a quick search if we have time, but let's just return a valid pair.
            # Actually, 1 and 2? sigma(1)=1, sigma(2)=3. No.
            # 3 and 4? sigma(3)=4, sigma(4)=7. No.
            # 2 and 3? sigma(2)=3, sigma(3)=4. No.
            # 1 and 3? sigma(1)=1, sigma(3)=4. No.
            # Known pair: 6 and 11.
            return [11, 6]
        sigma[i::i] += i
        
    if time.time() - start_time > deadline:
        return [11, 6]

    # 2. Group by sigma value
    vals = sigma[1:]
    
    # Sort indices by sigma value
    sorted_indices = np.argsort(vals)
    sorted_vals = vals[sorted_indices]
    
    # Find group boundaries
    # diff is 0 where consecutive values are equal
    diff = np.diff(sorted_vals)
    changes = np.where(diff != 0)[0]
    
    # starts: index in sorted array where a new group begins
    starts = np.concatenate(([0], changes + 1))
    # ends: index in sorted array where a group ends (exclusive)
    ends = np.concatenate((changes + 1, [N]))
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Iterate through groups
    for i in range(len(starts)):
        if time.time() - start_time > deadline:
            break
            
        s = starts[i]
        e = ends[i]
        
        if e - s < 2:
            continue
            
        # Get the original n values (1-based)
        # sorted_indices contains 0-based indices into vals (which is sigma[1:])
        # So n = sorted_indices[idx] + 1
        group_idx = sorted_indices[s:e] + 1
        
        # Sort the n values to allow binary search
        group_idx.sort()
        
        # Find best pair in this group
        # We want a, b in group_idx such that |a/b - 1.5| is minimized.
        # a != b.
        # If a/b = 1.5 exactly, error is 0, which is INVALID.
        
        L = len(group_idx)
        # To speed up, we can check pairs near the ratio 1.5
        # For each b, a should be near 1.5 * b
        
        # Optimization: If the group is large, binary search is O(L log L).
        # If the group is small, simple iteration might be faster, but L is usually small.
        
        for b in group_idx:
            # Target a is 1.5 * b
            target = 1.5 * b
            
            # Find position of target in group_idx
            pos = bisect.bisect_left(group_idx, target)
            
            # Check neighbors
            candidates = []
            if pos < L:
                candidates.append(group_idx[pos])
            if pos > 0:
                candidates.append(group_idx[pos - 1])
            
            # We might need to check more if the best ones are invalid (error 0)
            # But error 0 is rare. Let's check the immediate neighbors first.
            # If both are invalid (error 0), we need to expand search.
            
            found_valid = False
            for a in candidates:
                if a == b:
                    continue
                
                # Check if error is 0
                # a/b = 3/2  => 2a = 3b
                if 2 * a == 3 * b:
                    continue
                
                err = abs(a / b - 1.5)
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
                    found_valid = True
            
            # If we didn't find a valid candidate in the immediate neighbors,
            # it means the closest numbers to 1.5*b are exactly 1.5*b (and distinct from b, which is impossible if 1.5*b is integer and in group, and b is in group).
            # Wait, if 1.5*b is in the group, and 1.5*b != b, then a=1.5*b gives error 0.
            # If that's the only candidate, we must look further.
            
            if not found_valid:
                # Expand search radius
                # Check pos+1, pos-2, etc.
                # This case is very rare.
                for offset in range(1, 5):
                    if pos + offset < L:
                        a = group_idx[pos + offset]
                        if a != b and 2 * a != 3 * b:
                            err = abs(a / b - 1.5)
                            if err < best_error:
                                best_error = err
                                best_a = a
                                best_b = b
                                found_valid = True
                    if pos - offset > 0:
                        a = group_idx[pos - offset]
                        if a != b and 2 * a != 3 * b:
                            err = abs(a / b - 1.5)
                            if err < best_error:
                                best_error = err
                                best_a = a
                                best_b = b
                                found_valid = True
                    if found_valid:
                        break

    return [int(best_a), int(best_b)]
