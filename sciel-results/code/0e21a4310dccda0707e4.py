import numpy as np
import bisect
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    N = 10**7
    
    # 1. Compute sigma(n) for all n <= N using efficient sieve
    # Use linear sieve for SPF
    spf = np.zeros(N + 1, dtype=np.int64)
    primes = []
    spf[1] = 1
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
    
    # 2. Compute sigma using multiplicative property
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    
    for n in range(2, N + 1):
        if time.time() > deadline - 5.0:
            break
        p = spf[n]
        m = n // p
        # Count power of p in n
        pk = p
        while m % p == 0:
            pk *= p
            m //= p
        # sigma(n) = sigma(m) * (pk*p - 1) // (p - 1)
        sigma[n] = sigma[m] * (pk * p - 1) // (p - 1)
    
    if time.time() > deadline - 2.0:
        return [11, 6]
    
    # 3. Bucket indices by sigma value
    buckets = defaultdict(list)
    for i in range(1, N + 1):
        s = int(sigma[i])
        if s > 0:
            buckets[s].append(i)
    
    if time.time() > deadline - 1.0:
        for key, lst in buckets.items():
            if len(lst) >= 2:
                return [int(lst[0]), int(lst[1])]
        return [11, 6]
    
    # 4. Search for best ratio
    # Minimize |a/b - 3/2| = |2a - 3b| / (2b)
    
    best_a = 11
    best_b = 6
    best_num = abs(2 * 11 - 3 * 6)  # = 4
    best_den = 2 * 6  # = 12
    
    # Sort all bucket lists
    for key in buckets:
        if len(buckets[key]) >= 2:
            buckets[key].sort()
    
    # Process larger buckets first (more likely to contain good pairs)
    sorted_buckets = sorted(buckets.items(), key=lambda x: -len(x[1]))
    
    if time.time() > deadline - 0.5:
        return [int(best_a), int(best_b)]
    
    # Process buckets
    for sig_val, lst in sorted_buckets:
        if time.time() > deadline - 0.5:
            break
        if len(lst) < 2:
            continue
        
        # Use two-pointer approach: for each b, find a closest to 1.5*b
        # Since lst is sorted, we can use binary search efficiently
        for b in lst:
            if time.time() > deadline - 0.5:
                break
                
            # Compute range of a that could beat current best
            threshold = (best_num * b) // best_den + 1
            
            lo = max(1, (3 * b - threshold + 1) // 2)
            hi = min(N, (3 * b + threshold - 1) // 2)
            
            if lo > hi:
                continue
            
            left_idx = bisect.bisect_left(lst, lo)
            right_idx = bisect.bisect_right(lst, hi)
            
            # Only check a few candidates around 1.5*b
            target = 3 * b // 2
            center_idx = bisect.bisect_left(lst, target)
            
            # Check candidates near the center
            check_range = 20
            start_idx = max(left_idx, center_idx - check_range)
            end_idx = min(right_idx, center_idx + check_range)
            
            for idx in range(start_idx, end_idx):
                a = lst[idx]
                if a == b:
                    continue
                num = abs(2 * a - 3 * b)
                den = 2 * b
                if num * best_den < best_num * den:
                    best_a = a
                    best_b = b
                    best_num = num
                    best_den = den
    
    # Ensure result is valid (not exact 3/2)
    if best_num == 0:
        best_a, best_b = 11, 6
        best_num, best_den = 4, 12
    
    return [int(best_a), int(best_b)]
