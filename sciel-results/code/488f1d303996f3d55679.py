import time
import math
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin
    
    N = 10_000_000
    
    # Use Python lists for speed in the tight loop
    spf = [0] * (N + 1)
    spf[0] = 0
    spf[1] = 1
    
    # Sieve for smallest prime factor
    # Optimization: only iterate up to sqrt(N) for the inner loop start
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= N:
                # Mark multiples
                # We can use slicing for assignment if we were setting to i, but we only set if 0.
                # So we must loop.
                for j in range(i * i, N + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
                        
    if time.time() - start_time > deadline - 40:
        return [11, 6]
        
    sigma = [0] * (N + 1)
    sigma[1] = 1
    rem = [0] * (N + 1)
    pfac = [0] * (N + 1)
    rem[1] = 1
    pfac[1] = 1
    
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            rem_n = rem[m]
            pfac_n = pfac[m] * p
            rem[n] = rem_n
            pfac[n] = pfac_n
            # sigma[n] = sigma[rem_n] * (pfac_n * p - 1) // (p - 1)
            sigma[n] = sigma[rem_n] * ((pfac_n * p - 1) // (p - 1))
        else:
            rem[n] = m
            pfac[n] = p
            sigma[n] = sigma[m] * (p + 1)
            
    if time.time() - start_time > deadline - 30:
        return [11, 6]
        
    # Group by sigma
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[sigma[n]].append(n)
        
    if time.time() - start_time > deadline - 15:
        return [11, 6]
        
    best_error = float('inf')
    best_a = 11
    best_b = 6
    early_exit_threshold = 1e-9 # Try to find even better
    
    # Sort groups by size descending to find good candidates faster?
    # Actually, smaller groups are faster to process, but less likely to have good pairs.
    # Large groups are more likely to have pairs close to 1.5.
    # Let's iterate all.
    
    for s_val, nums in groups.items():
        if time.time() - start_time > deadline:
            break
            
        if len(nums) < 2:
            continue
            
        nums.sort()
        
        # Binary search for each b
        # We want a such that a/b ~ 1.5 => a ~ 1.5 b
        for b in nums:
            if time.time() - start_time > deadline:
                break
                
            target = 1.5 * b
            idx = bisect.bisect_left(nums, target)
            
            # Check idx and idx-1
            if idx < len(nums):
                a = nums[idx]
                if a != b:
                    if 2 * a != 3 * b:
                        num = abs(2 * a - 3 * b)
                        den = 2 * b
                        err = num / den
                        if err < best_error:
                            best_error = err
                            best_a = a
                            best_b = b
                            if best_error < early_exit_threshold:
                                return [int(best_a), int(best_b)]
            if idx > 0:
                a = nums[idx - 1]
                if a != b:
                    if 2 * a != 3 * b:
                        num = abs(2 * a - 3 * b)
                        den = 2 * b
                        err = num / den
                        if err < best_error:
                            best_error = err
                            best_a = a
                            best_b = b
                            if best_error < early_exit_threshold:
                                return [int(best_a), int(best_b)]
                                
    if best_error == float('inf'):
        return [11, 6]
        
    return [int(best_a), int(best_b)]
