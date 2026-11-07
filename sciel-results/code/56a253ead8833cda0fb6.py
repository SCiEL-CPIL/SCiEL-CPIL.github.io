"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX = 10000000
    
    # 1. Sieve for smallest prime factor
    spf = list(range(MAX + 1))
    for i in range(2, int(MAX**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    if time.time() > deadline:
        return [11, 6]

    # 2. Compute sigma
    sigma = [0] * (MAX + 1)
    power = [0] * (MAX + 1)
    sigma[1] = 1
    power[1] = 1
    
    for i in range(2, MAX + 1):
        p = spf[i]
        m = i // p
        if m % p == 0:
            power[i] = power[m] * p
            r = m // power[m]
            sigma[i] = sigma[m] + power[i] * sigma[r]
        else:
            power[i] = p
            sigma[i] = sigma[m] * (p + 1)
            
        if time.time() > deadline:
            return [11, 6]

    # 3. Group numbers by sigma value
    groups = {}
    for i in range(1, MAX + 1):
        s = sigma[i]
        if s in groups:
            groups[s].append(i)
        else:
            groups[s] = [i]
            
    if time.time() > deadline:
        return [11, 6]

    # 4. Search for best pair
    best_a = 11
    best_b = 6
    best_num = 4  # |2*11 - 3*6|
    best_den = 12 # 2*6
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        
        nums.sort()
        
        for b in nums:
            target_a = 1.5 * b
            
            # Find insertion point
            idx = bisect.bisect_left(nums, target_a)
            
            candidates = []
            if idx < len(nums):
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx-1])
                
            for a in candidates:
                if a == b:
                    continue
                diff = abs(2 * a - 3 * b)
                den = 2 * b
                if diff * best_den < best_num * den:
                    best_num = diff
                    best_den = den
                    best_a = a
                    best_b = b
                    
        if time.time() > deadline:
            break

    return [int(best_a), int(best_b)]
# EVOLVE-BLOCK-END
