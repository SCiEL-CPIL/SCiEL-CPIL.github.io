import time
from fractions import Fraction
from collections import defaultdict

def construct():
    """
    Find positive integers a, b <= 10^7 such that sigma(a) = sigma(b)
    minimizing |a/b - 3/2|.
    """
    deadline = time.time() + 55
    MAX_N = 10000000
    
    # Compute smallest prime factor
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute sigma(n) using the multiplicative property
    # sigma(n) = product over prime powers p^k || n of (p^(k+1)-1)/(p-1)
    # We'll compute it incrementally:
    # Let n = p^k * m where p = spf(n) and p does not divide m
    # Then sigma(n) = sigma(m) * (1 + p + ... + p^k)
    
    # First, compute the highest power of spf[n] dividing n
    # p_power[n] = p^k where p^k || n and p = spf[n]
    p_power = [1] * (MAX_N + 1)
    for i in range(2, MAX_N + 1):
        p = spf[i]
        m = i // p
        if m % p == 0:
            p_power[i] = p_power[m] * p
        else:
            p_power[i] = p
    
    # Now compute sigma
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    for i in range(2, MAX_N + 1):
        p = spf[i]
        pk = p_power[i]
        r = i // pk  # r is coprime to p
        # (1 + p + ... + p^k) where p^k = pk
        # = (p^(k+1) - 1) / (p - 1) = (pk * p - 1) / (p - 1)
        if p == 2:
            psum = 2 * pk - 1
        else:
            psum = (pk * p - 1) // (p - 1)
        sigma[i] = sigma[r] * psum
    
    # Group numbers by their sigma value
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        groups[sigma[n]].append(n)
    
    # Find pairs (a, b) in the same group minimizing |a/b - 3/2|
    best_error = Fraction(1, 1)  # Initial large error
    best_a, best_b = 1, 1
    
    target = Fraction(3, 2)
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        if time.time() > deadline - 10:
            break
        
        nums.sort()
        
        # For each b, find a close to 3b/2
        for b in nums:
            target_a = (3 * b) / 2  # as float for binary search
            # Binary search in nums for value closest to target_a
            lo, hi = 0, len(nums) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if nums[mid] * 2 == 3 * b:
                    a = nums[mid]
                    err = abs(Fraction(a, b) - target)
                    if err < best_error:
                        best_error = err
                        best_a, best_b = a, b
                    break
                elif nums[mid] * 2 < 3 * b:
                    lo = mid + 1
                else:
                    hi = mid - 1
            
            # Check the closest candidates
            for idx in [lo, lo - 1, hi, hi + 1]:
                if 0 <= idx < len(nums):
                    a = nums[idx]
                    err = abs(Fraction(a, b) - target)
                    if err < best_error:
                        best_error = err
                        best_a, best_b = a, b
    
    return [int(best_a), int(best_b)]
