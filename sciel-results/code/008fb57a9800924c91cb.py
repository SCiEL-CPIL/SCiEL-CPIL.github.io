import numpy as np
import time
from collections import defaultdict

def construct():
    start = time.time()
    deadline = start + 88  # Safety margin before 90s limit

    # Phase 1: Compute sigma up to N using optimized NumPy sieve
    N = 10_000_000
    sigma = np.zeros(N + 1, dtype=np.int64)
    # Only need to go up to N/2 for the sieve (larger i contribute only to sigma[i] itself)
    for i in range(1, N // 2 + 1):
        sigma[i::i] += i
    # Add self-contribution for all numbers
    sigma[1:] += np.arange(1, N + 1, dtype=np.int64)

    if time.time() > deadline - 30:
        return [11, 6]

    best_a, best_b = 11, 6
    best_num = abs(2*11 - 3*6)  # = 4
    best_den = 2*6  # = 12

    # Phase 2: Group numbers by sigma value, then for each group find pairs close to ratio 3/2
    # Build dictionary: sigma_value -> list of numbers
    # To save memory/time, only consider numbers up to N
    sigma_vals = sigma[1:N+1]
    
    # Use argsort to group
    sorted_indices = np.argsort(sigma_vals, kind='stable')
    sorted_sigmas = sigma_vals[sorted_indices]
    
    # Find group boundaries
    change_points = np.concatenate(([0], np.where(np.diff(sorted_sigmas) != 0)[0] + 1, [len(sorted_sigmas)]))
    
    for idx in range(len(change_points) - 1):
        if time.time() > deadline - 15:
            break
        s_start = change_points[idx]
        s_end = change_points[idx + 1]
        if s_end - s_start < 2:
            continue
        
        nums = sorted_indices[s_start:s_end] + 1  # Convert 0-indexed to 1-indexed
        nlen = len(nums)
        
        # For each b in this group, find a in same group closest to 3b/2
        # Use binary search on the sorted nums
        for k in range(nlen):
            if time.time() > deadline - 15:
                break
            b = int(nums[k])
            target_a = (3 * b) // 2
            # Binary search for target_a in nums
            lo, hi = 0, nlen - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            # Check lo and lo-1
            for idx2 in (lo, lo - 1):
                if 0 <= idx2 < nlen:
                    a = int(nums[idx2])
                    if a == b:
                        continue
                    num = abs(2*a - 3*b)
                    den = 2*b
                    if num * best_den < best_num * den:
                        best_num = num
                        best_den = den
                        best_a, best_b = a, b

    # Phase 3: Targeted search using ratio constraint 2a - 3b = k
    # For small |k|, the error |a/b - 3/2| = |k|/(2b) is small when b is large.
    # Vectorized check for each k
    if time.time() < deadline - 5:
        for k in range(-100, 101):
            if time.time() > deadline - 5:
                break
            # a = (3b + k) / 2 must be integer, so 3b + k must be even
            # 3b + k even => b must have same parity as k (since 3b has same parity as b)
            if k % 2 != 0:
                # b must be odd
                min_b = 1
                if min_b % 2 == 0:
                    min_b += 1
            else:
                # b must be even
                min_b = 2
                if min_b % 2 == 0:
                    min_b += 0  # already even
            # b range: a = (3b+k)/2 <= 10^7 => b <= (20000000-k)/3
            # a >= 1 => 3b+k >= 2 => b >= ceil((2-k)/3)
            max_b = min(10_000_000, (20_000_000 - k) // 3)
            min_b = max(min_b, (2 - k + 2) // 3)  # ceil((2-k)/3)
            
            if min_b > max_b:
                continue
            
            # Create array of b values with correct parity
            b_vals = np.arange(min_b, max_b + 1, 2, dtype=np.int64)
            a_vals = (3 * b_vals + k) // 2
            
            # Filter valid a values
            valid = (a_vals >= 1) & (a_vals <= N) & (a_vals != b_vals)
            if not np.any(valid):
                continue
            b_vals = b_vals[valid]
            a_vals = a_vals[valid]
            
            # Check sigma equality vectorized
            sigma_a = sigma[a_vals]
            sigma_b = sigma[b_vals]
            matches = np.where(sigma_a == sigma_b)[0]
            
            for m in matches:
                a = int(a_vals[m])
                b = int(b_vals[m])
                num = abs(2*a - 3*b)
                den = 2*b
                if num * best_den < best_num * den:
                    best_num = num
                    best_den = den
                    best_a, best_b = a, b

    return [int(best_a), int(best_b)]
