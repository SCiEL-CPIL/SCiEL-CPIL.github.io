import numpy as np
import bisect
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    N = 10**7
    
    # 1. Compute sigma(n) for all n <= N using linear sieve for SPF
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
        if time.time() > deadline - 10.0:
            break
        p = spf[n]
        m = n // p
        pk = p
        while m % p == 0:
            pk *= p
            m //= p
        sigma[n] = sigma[m] * (pk * p - 1) // (p - 1)
    
    if time.time() > deadline - 5.0:
        return [11, 6]
    
    # 3. Bucket indices by sigma value
    buckets = defaultdict(list)
    for i in range(1, N + 1):
        s = int(sigma[i])
        if s > 0:
            buckets[s].append(i)
    
    if time.time() > deadline - 3.0:
        for key, lst in buckets.items():
            if len(lst) >= 2:
                return [int(lst[0]), int(lst[1])]
        return [11, 6]
    
    # 4. Sort all bucket lists and keep only those with >= 2 elements
    for key in buckets:
        if len(buckets[key]) >= 2:
            buckets[key].sort()
    
    sorted_buckets = [(k, v) for k, v in buckets.items() if len(v) >= 2]
    sorted_buckets.sort(key=lambda x: -len(x[1]))
    
    if time.time() > deadline - 1.0:
        return [11, 6]
    
    # 5. Search for best ratio: minimize |a/b - 3/2| = |2a - 3b| / (2b)
    best_a = 11
    best_b = 6
    best_num = 4
    best_den = 12
    
    for sig_val, lst in sorted_buckets:
        if time.time() > deadline - 0.5:
            break
        if len(lst) < 2:
            continue
        
        # Adaptive check range: scale with bucket size
        check_range = min(100, max(10, len(lst) // 5))
        
        for a in lst:
            if time.time() > deadline - 0.5:
                break
            
            # Ideal b = 2a/3
            target_b = (2 * a) // 3
            
            center_idx = bisect.bisect_left(lst, target_b)
            
            start_idx = max(0, center_idx - check_range)
            end_idx = min(len(lst), center_idx + check_range)
            
            for idx in range(start_idx, end_idx):
                b = lst[idx]
                if a == b:
                    continue
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                den = 2 * b
                if num * best_den < best_num * den:
                    best_a = a
                    best_b = b
                    best_num = num
                    best_den = den
                    if num == 1:
                        return [int(best_a), int(best_b)]
    
    if best_num == 0:
        best_a, best_b = 11, 6
        best_num, best_den = 4, 12
    
    return [int(best_a), int(best_b)]
