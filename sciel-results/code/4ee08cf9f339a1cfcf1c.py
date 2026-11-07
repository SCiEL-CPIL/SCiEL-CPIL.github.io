import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin within 90s
    
    N = 10_000_000
    
    # Step 1: Build smallest prime factor sieve (O(N log log N))
    spf = np.zeros(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 40:
        return [11, 6]
    
    # Step 2: Compute sigma(n) multiplicatively in O(N)
    # sigma(n) = product over prime factors p^k || n of (1 + p + ... + p^k)
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    # p_power_sigma[n] = (1 + p + ... + p^k) where p^k is the highest power of spf[n] dividing n
    p_power_sigma = np.zeros(N + 1, dtype=np.int64)
    
    for n in range(2, N + 1):
        if time.time() - start_time > deadline - 35:
            break
        p = spf[n]
        m = n // p
        if m % p == 0:
            # p divides m, so p^(k+1) | n for some k >= 1
            # p_power_sigma[m] = 1 + p + ... + p^k
            # p_power_sigma[n] = 1 + p + ... + p^(k+1) = p_power_sigma[m] + p^(k+1)
            # We need p^(k+1) = p * p^k
            # But we don't directly store p^k. However:
            # p_power_sigma[n] = p_power_sigma[m] + p * (p_power_sigma[m] - p_power_sigma[m//p]) ... no
            # Simpler: p_power_sigma[n] = p_power_sigma[m] + p^(k+1)
            # And p^(k+1) = p * p^k where p^k is highest power of p dividing m
            # Let's track p_power separately or compute differently
            # Actually: if m = p^k * r with gcd(p,r)=1, then
            # p_power_sigma[m] = 1 + p + ... + p^k
            # p_power_sigma[n] = 1 + p + ... + p^(k+1) = p_power_sigma[m] + p^(k+1)
            # p^(k+1) = p * p^k
            # We can compute p^k as: while dividing m by p... but that's expensive
            # Alternative: store the actual power
            pass
        else:
            # p does not divide m, so p || n (p^1)
            # p_power_sigma[n] = 1 + p
            # sigma[n] = sigma[m] * (1 + p)
            p_power_sigma[n] = 1 + p
            sigma[n] = sigma[m] * (1 + p)
    
    # The multiplicative approach with the "p divides m" case is complex.
    # Let me use a cleaner O(N) approach:
    # For each n, find its prime factorization using spf, compute sigma.
    # But doing this for all n is O(N * number_of_prime_factors) which is fine.
    
    # Actually, let me use a different O(N) method:
    # sigma is multiplicative. Use the recurrence:
    # If n = p^k * m where gcd(p,m)=1 and p=spf[n]:
    #   if k=1: sigma[n] = sigma[m] * (1+p), ppow[n] = 1+p
    #   if k>1: let m' = m*p^(k-1) = n/p. Then m' has spf=p and ppow[m'] = 1+p+...+p^(k-1)
    #           ppow[n] = ppow[m'] + p^k
    #           sigma[n] = sigma[n/p] * ppow[n] / ppow[m']
    # This still requires tracking powers.
    
    # Simplest correct O(N) approach: just factor each number using spf
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    for n in range(2, N + 1):
        if time.time() - start_time > deadline - 30:
            break
        orig = n
        result = 1
        while n > 1:
            p = spf[n]
            # Sum of powers: 1 + p + p^2 + ... + p^k
            term = 1
            power = 1
            while n % p == 0:
                n //= p
                power *= p
                term += power
            result *= term
        sigma[orig] = result
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return [11, 6]
    
    # Step 3: Group by sigma value
    # Use a dict mapping sigma -> list of n
    groups = defaultdict(list)
    for n in range(1, N + 1):
        if time.time() - start_time > deadline - 15:
            break
        s = int(sigma[n])
        groups[s].append(n)
    
    # Step 4: Search for best pair
    # Sort groups by size descending - larger groups more likely to have close pairs
    group_list = [(len(v), k, v) for k, v in groups.items() if len(v) >= 2]
    group_list.sort(key=lambda x: -x[0])
    
    best_num = 0  # numerator of |2a - 3b|
    best_den = 0  # denominator = 2b
    best_a = 11
    best_b = 6
    
    # Process groups, prioritizing larger ones
    processed = 0
    for size, s_val, nums in group_list:
        if time.time() - start_time > deadline - 5:
            break
        processed += 1
        
        nums.sort()
        n_len = len(nums)
        
        # For each b, find a closest to 1.5*b
        # Two-pointer approach: since we want a/b ≈ 3/2, a ≈ 1.5b
        # For sorted nums, use two pointers
        # Actually, for each b, binary search for 1.5*b
        for i in range(n_len):
            b = nums[i]
            target = (3 * b) // 2  # floor of 1.5*b
            # Binary search for target in nums
            lo = 0
            hi = n_len
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            # Check lo and lo-1
            for j in [lo, lo - 1]:
                if 0 <= j < n_len and j != i:
                    a = nums[j]
                    # Compute |a/b - 3/2| = |2a - 3b| / (2b)
                    # Exact integer comparison
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    if num == 0:
                        continue  # Invalid, exact ratio
                    # Compare num/den with best_num/best_den
                    # num/best_den < best_num/den  <=>  num*den < best_num*best_den
                    if best_den == 0 or num * best_den < best_num * den:
                        best_num = num
                        best_den = den
                        best_a = a
                        best_b = b
    
    # Fallback
    if best_a == 11 and best_b == 6:
        # Verify this is valid
        # sigma(11) = 1+11 = 12, sigma(6) = 1+2+3+6 = 12. Yes!
        pass
    
    return [int(best_a), int(best_b)]
