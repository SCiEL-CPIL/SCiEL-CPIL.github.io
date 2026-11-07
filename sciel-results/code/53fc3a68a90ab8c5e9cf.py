import sys
import time
import math
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
MAX_N = 10000000

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Compute smallest prime factor (SPF) sieve
    spf = np.zeros(MAX_N + 1, dtype=np.int64)
    for i in range(2, MAX_N + 1):
        if spf[i] == 0:
            for j in range(i, MAX_N + 1, i):
                if spf[j] == 0:
                    spf[j] = i
    
    # 2. Compute sigma(n) for all n using SPF
    sigma = np.ones(MAX_N + 1, dtype=np.int64)
    for n in range(2, MAX_N + 1):
        if sigma[n] == 1:  # n is prime
            sigma[n] = n + 1
        else:
            p = spf[n]
            m = n // p
            # Count power of p in n
            pk = p
            while m % p == 0:
                m //= p
                pk *= p
            # sigma(n) = sigma(m) * (1 + p + p^2 + ... + p^k)
            # where pk = p^k, and m is coprime to p
            sum_pk = (pk * p - 1) // (p - 1) if p > 1 else pk
            sigma[n] = sigma[m] * sum_pk
    
    # 3. Group numbers by sigma value
    sigma_to_nums = {}
    for n in range(1, MAX_N + 1):
        s = int(sigma[n])
        if s not in sigma_to_nums:
            sigma_to_nums[s] = []
        sigma_to_nums[s].append(n)
    
    # 4. Search for best pair (a, b) with sigma(a) = sigma(b)
    best_a, best_b = 1, 1
    best_err = float('inf')
    
    # For each sigma value, check all pairs in that group
    # To minimize |a/b - 3/2|, we want a/b ≈ 3/2, so a ≈ 1.5*b
    # Sort each group and use two pointers or binary search
    
    for s, nums in sigma_to_nums.items():
        if len(nums) < 2:
            continue
        
        nums.sort()
        n = len(nums)
        
        # For each b in nums, find a in nums such that a/b ≈ 3/2
        # a ≈ 3b/2
        # Use binary search or two-pointer approach
        
        # Two-pointer: for each b, we want a close to 3b/2
        # Since nums is sorted, we can use bisect
        
        import bisect
        for i in range(n):
            b = nums[i]
            target_a = 3 * b // 2
            
            # Find closest elements to target_a in nums
            # Use bisect to find insertion point
            pos = bisect.bisect_left(nums, target_a)
            
            # Check candidates around pos
            candidates = []
            if pos < n:
                candidates.append(nums[pos])
            if pos > 0:
                candidates.append(nums[pos - 1])
            
            for a in candidates:
                if a == b:
                    continue
                # Compute |a/b - 3/2| = |2a - 3b| / (2b)
                err = abs(2 * a - 3 * b) / (2.0 * b)
                if err < best_err:
                    best_err = err
                    best_a, best_b = a, b
            
            # Time check
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
    
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
