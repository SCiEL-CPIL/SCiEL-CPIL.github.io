import numpy as np

def construct():
    N = 10000000
    
    # Compute smallest prime factor (SPF) sieve
    spf = np.arange(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute sigma(n) for all n
    sigma = np.ones(N + 1, dtype=np.int64)
    for n in range(2, N + 1):
        if spf[n] == n:
            # n is prime
            sigma[n] = n + 1
        else:
            # Factorize n using spf
            temp = n
            result = 1
            while temp > 1:
                p = spf[temp]
                # Count power of p
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                # sigma(p^k) = (p^(k+1) - 1) / (p - 1)
                result *= (power * p - 1) // (p - 1)
            sigma[n] = result
    
    # Group numbers by sigma value
    from collections import defaultdict
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[sigma[n]].append(n)
    
    # For each group, find pair (a,b) minimizing |a/b - 3/2|
    best_error = float('inf')
    best_pair = (1, 1)
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        nums.sort()
        # For each b in nums, find a closest to 1.5*b
        for i in range(len(nums)):
            b = nums[i]
            target_a = 1.5 * b
            # Binary search for closest a
            lo, hi = 0, len(nums) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            # Check lo and lo-1
            for idx in [lo, lo - 1]:
                if 0 <= idx < len(nums):
                    a = nums[idx]
                    if a == b:
                        continue
                    # Compute |a/b - 3/2| = |2a - 3b| / (2b)
                    diff = abs(2 * a - 3 * b)
                    if diff == 0:
                        continue  # Invalid, ratio exactly 3/2
                    error = diff / (2.0 * b)
                    if error < best_error:
                        best_error = error
                        best_pair = (a, b)
    
    return [best_pair[0], best_pair[1]]
