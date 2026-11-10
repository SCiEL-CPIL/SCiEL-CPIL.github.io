import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Strategy: compute sigma for numbers up to N using a sieve
    # Then for each sigma value, find pairs (a,b) with a/b closest to 3/2
    
    N = 2_000_000  # Start with a reasonable limit
    
    # Sieve for sigma
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Build dictionary: sigma value -> list of n with that sigma
    from collections import defaultdict
    sigma_groups = defaultdict(list)
    for n in range(1, N + 1):
        sigma_groups[sigma[n]].append(n)
    
    # For each group with at least 2 elements, find best pair with ratio near 1.5
    best_a, best_b = 11, 6  # fallback
    best_num = abs(2*11 - 3*6)  # |22-18| = 4
    best_den = 2*6  # 12
    
    checked = 0
    for s, nums in sigma_groups.items():
        if time.time() > deadline - 10:
            break
        if len(nums) < 2:
            continue
        nums.sort()
        nlen = len(nums)
        checked += 1
        
        # For each b in nums, find a in nums closest to 1.5*b
        for k in range(nlen):
            b = nums[k]
            target_a = int(1.5 * b)
            # Binary search for target_a in nums
            lo, hi = 0, nlen - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            # Check lo and lo-1
            for idx in (lo, lo - 1):
                if 0 <= idx < nlen:
                    a = nums[idx]
                    if a == b:
                        continue
                    num = abs(2*a - 3*b)
                    den = 2*b
                    if num * best_den < best_num * den:
                        best_num = num
                        best_den = den
                        best_a, best_b = a, b
    
    return [int(best_a), int(best_b)]
