import numpy as np
import time
import math
from collections import defaultdict
import bisect

def construct():
    t_start = time.time()
    deadline = t_start + 50.0  # Safety margin for 60s limit
    
    MAX_N = 10_000_000
    
    # 1. Linear Sieve for Sigma
    # Arrays
    lp = [0] * (MAX_N + 1)       # lowest prime
    pow_lp = [0] * (MAX_N + 1)   # highest power of lp dividing n
    h = [0] * (MAX_N + 1)        # sum of powers of lp: 1 + p + ... + p^k
    sigma = [0] * (MAX_N + 1)    # sigma(n)
    primes = []
    
    sigma[1] = 1
    h[1] = 1
    pow_lp[1] = 1
    
    for i in range(2, MAX_N + 1):
        if lp[i] == 0:
            lp[i] = i
            primes.append(i)
            sigma[i] = i + 1
            h[i] = i + 1
            pow_lp[i] = i
            
        for p in primes:
            if p > lp[i] or i * p > MAX_N:
                break
            lp[i * p] = p
            
            if p == lp[i]:
                # p divides i
                # i = p^k * m
                # i*p = p^(k+1) * m
                # h[i] = 1 + ... + p^k
                # h[i*p] = h[i] + p^(k+1) = h[i] + pow_lp[i] * p
                # pow_lp[i*p] = pow_lp[i] * p
                # sigma[i*p] = sigma[m] * h[i*p]
                # m = i // pow_lp[i]
                m = i // pow_lp[i]
                pow_lp[i * p] = pow_lp[i] * p
                h[i * p] = h[i] + pow_lp[i] * p
                sigma[i * p] = sigma[m] * h[i * p]
                break
            else:
                # p < lp[i], so gcd(i, p) = 1
                # sigma[i*p] = sigma[i] * sigma[p] = sigma[i] * (p + 1)
                # h[i*p] = 1 + p
                # pow_lp[i*p] = p
                pow_lp[i * p] = p
                h[i * p] = p + 1
                sigma[i * p] = sigma[i] * (p + 1)
                
    if time.time() > deadline - 5:
        return [6, 11]

    # 2. Group by sigma value
    buckets = defaultdict(list)
    # Use local variables for speed
    sig = sigma
    for i in range(1, MAX_N + 1):
        buckets[sig[i]].append(i)
        
    if time.time() > deadline - 5:
        return [6, 11]

    # 3. Find best pair
    best_err = float('inf')
    best_pair = (6, 11) # Fallback
    
    # Optimization: Pre-sort lists?
    # The search requires sorted lists for binary search.
    # Sorting all lists takes sum(len(vals) log len(vals)).
    # Total elements is N.
    # This is acceptable.
    
    for s, vals in buckets.items():
        if len(vals) < 2:
            continue
            
        vals.sort()
        
        for b in vals:
            if time.time() > deadline - 2:
                return [int(best_pair[0]), int(best_pair[1])]
                
            target = 1.5 * b
            idx = bisect.bisect_left(vals, target)
            
            candidates = []
            if idx < len(vals):
                candidates.append(vals[idx])
            if idx > 0:
                candidates.append(vals[idx-1])
                
            for a in candidates:
                if a == b:
                    continue
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue
                err = diff / (2.0 * b)
                if err < best_err:
                    best_err = err
                    best_pair = (a, b)
                    
    return [int(best_pair[0]), int(best_pair[1])]
