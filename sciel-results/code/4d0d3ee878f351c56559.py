import time
import math
import numpy as np
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for all n <= N using SPF-based multiplicative approach
    # sigma is multiplicative: if n = prod p_i^e_i, then sigma(n) = prod (p_i^(e_i+1)-1)/(p_i-1)
    
    # Step 1a: Compute smallest prime factor (SPF) for all numbers up to N
    spf = np.zeros(N + 1, dtype=np.int32)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
    
    if time.time() - start_time > deadline - 40:
        return [11, 6]
    
    # Step 1b: Compute sigma(n) using multiplicative property
    # For each n, factorize using spf and compute sigma
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    
    # For each prime p, process all numbers divisible by p
    # We'll use the fact that sigma is multiplicative
    # Process by prime factors: for each prime p, for each power p^k,
    # update sigma for numbers that have p^k as their highest power of p
    
    # Alternative: iterate through all numbers and factorize
    # But that's O(N log N) worst case. Let's use a more efficient approach.
    
    # For each n from 2 to N:
    #   let p = spf[n], m = n // p
    #   if m % p == 0: n = p^k * r where p doesn't divide r
    #     sigma(n) = sigma(r) * (p^(k+1)-1)/(p-1)
    #   else: n = p * r where p doesn't divide r
    #     sigma(n) = sigma(r) * (p+1)
    
    # We need to track the power of the smallest prime factor
    # Let's use arrays:
    #   spf_power[n] = highest power of spf[n] dividing n
    #   sigma[n] computed incrementally
    
    # Actually, let's use a cleaner approach:
    # For each n, write n = p^k * m where p = spf[n] and gcd(m, p) = 1
    # sigma(n) = sigma(m) * (p^(k+1) - 1) / (p - 1)
    
    # We can compute this iteratively. Let's define:
    #   rem[n] = n with all factors of spf[n] removed (i.e., m above)
    #   pwr[n] = p^k (the power part)
    
    # But storing these for 10M numbers is memory-intensive. Let's try direct factorization.
    
    # More efficient: use the recurrence
    # Let p = spf[n], m = n // p
    # If m % p != 0: sigma[n] = sigma[m] * (p + 1)
    # If m % p == 0: let m = p^j * r, then sigma[n] = sigma[r] * (p^(j+2)-1)/(p-1)
    #   But we need to know j. 
    
    # Let's use two arrays:
    #   rem[n] = n with all factors of spf[n] removed
    #   pfac[n] = spf[n]^k where k is the exponent of spf[n] in n
    
    rem = np.zeros(N + 1, dtype=np.int32)
    pfac = np.zeros(N + 1, dtype=np.int64)
    rem[1] = 1
    pfac[1] = 1
    
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # n = p * m, and p divides m
            # m = p^j * r, so n = p^(j+1) * r
            # rem[n] = rem[m] = r
            # pfac[n] = pfac[m] * p = p^(j+1)
            rem[n] = rem[m]
            pfac[n] = pfac[m] * p
            # sigma[n] = sigma[rem[n]] * (pfac[n]*p - 1) / (p - 1)
            # = sigma[rem[n]] * (p^(j+2) - 1) / (p - 1)
            # But we need sigma[rem[n]]... this gets complex.
            
            # Alternative: sigma[n] = sigma[m] + p^j * sigma[r]... no.
            # Let's just compute directly:
            # sigma[n] = sigma[r] * (1 + p + p^2 + ... + p^(j+1))
            # where r = rem[n], j+1 is the exponent
            # sum = (p^(j+2) - 1) / (p - 1)
            # But we need sigma[r]. We can compute it later or store it.
            
            # Actually, let's use a different approach. Since we're iterating in order,
            # and rem[n] < n, we can have sigma[rem[n]] already computed if we process carefully.
            # But rem[n] might not be less than n in the iteration order... actually it is,
            # since rem[n] = n / p^(j+1) < n.
            
            # So sigma[rem[n]] should be computed already.
            # sigma[n] = sigma[rem[n]] * (p^(j+2) - 1) // (p - 1)
            # where j+1 is such that pfac[n] = p^(j+1)
            # p^(j+2) = pfac[n] * p
            # sum = (pfac[n] * p - 1) // (p - 1)
            sigma[n] = sigma[rem[n]] * ((pfac[n] * p - 1) // (p - 1))
        else:
            # n = p * m, p does not divide m
            # rem[n] = m, pfac[n] = p
            rem[n] = m
            pfac[n] = p
            # sigma[n] = sigma[m] * (1 + p)
            sigma[n] = sigma[m] * (p + 1)
    
    if time.time() - start_time > deadline - 30:
        return [11, 6]
    
    # 2. Group numbers by their sigma value
    # Using a dictionary of lists
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[int(sigma[n])].append(n)
        
    if time.time() - start_time > deadline - 15:
        return [11, 6]
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    # Early exit if we get a very good answer
    early_exit_threshold = 1e-8
    
    # 3. Search for pairs
    # We want to minimize |a/b - 1.5| subject to sigma(a)=sigma(b) and a!=b (if a/b=1.5 exactly is invalid? 
    # "A raw value of exactly 0 cannot be scored and counts as invalid.")
    # So if 2a == 3b, we must skip that pair.
    
    # Sort groups by size descending to find good candidates faster?
    # Or just iterate all.
    # Large groups have more chances of having close ratios.
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
    
    for s_val, nums in sorted_groups:
        if time.time() - start_time > deadline:
            break
            
        if len(nums) < 2:
            continue
            
        nums.sort()
        
        # For each b in nums, find a in nums closest to 1.5 * b
        # We can optimize by noticing that if we iterate b, a is determined.
        # But doing binary search for every b is O(K log K).
        # For K up to a few thousand, this is fine.
        
        for b in nums:
            if time.time() - start_time > deadline:
                break
                
            target = 1.5 * b
            # Find position where target would be inserted
            idx = bisect.bisect_left(nums, target)
            
            # Check idx and idx-1
            for j in [idx, idx - 1]:
                if 0 <= j < len(nums):
                    a = nums[j]
                    if a == b:
                        continue
                    # Check if ratio is exactly 1.5
                    if 2 * a == 3 * b:
                        continue
                    
                    # Exact rational comparison: |2a-3b|/(2b) < best_error
                    # => |2a-3b| < best_error * 2b
                    # But best_error is float, so let's just compute and compare
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    err = num / den
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
                        if best_error < early_exit_threshold:
                            # Very good, return immediately
                            return [int(best_a), int(best_b)]
                        
    if best_error == float('inf'):
        return [11, 6]
        
    return [int(best_a), int(best_b)]
