def construct():
    import math
    from array import array
    from collections import defaultdict
    import bisect

    MAX_N = 10_000_000
    
    # 1. Linear Sieve for Smallest Prime Factor (SPF)
    spf = array('I', [0]) * (MAX_N + 1)
    primes = []
    for i in range(2, MAX_N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > MAX_N:
                break
            spf[i * p] = p

    # 2. Compute sigma for all n
    sigma = array('Q', [0]) * (MAX_N + 1)
    sigma[1] = 1
    for n in range(2, MAX_N + 1):
        orig_n = n
        s = 1
        while n > 1:
            p = spf[n]
            # Calculate sum of powers of p
            power_sum = 1
            power = 1
            while n % p == 0:
                n //= p
                power *= p
                power_sum += power
            s *= power_sum
        sigma[orig_n] = s

    # 3. Group numbers by sigma
    sigma_groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_groups[sigma[n]].append(n)
        
    # 4. Find best pair
    best_error = float('inf')
    best_a, best_b = None, None
    best_num = 0
    best_den = 1
    
    for sig_val, nums in sigma_groups.items():
        if len(nums) < 2:
            continue
        
        nums.sort()
        k = len(nums)
        
        # For each b in nums, find a in nums closest to 1.5*b
        for i in range(k):
            b = nums[i]
            target = 1.5 * b
            
            # Binary search for target in nums
            pos = bisect.bisect_left(nums, target)
            
            candidates = []
            if pos < k:
                candidates.append(nums[pos])
            if pos > 0:
                candidates.append(nums[pos-1])
                
            for a in candidates:
                if a == b:
                    continue
                # Check error |a/b - 1.5| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                den = 2 * b
                
                # Compare num/den with best_num/best_den
                if best_a is None or num * best_den < best_num * den:
                    best_a = a
                    best_b = b
                    best_num = num
                    best_den = den
                    
    if best_a is None:
        return [11, 6]
        
    return [best_a, best_b]
