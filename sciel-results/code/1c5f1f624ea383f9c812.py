import time
import math
from collections import defaultdict
import bisect

def construct():
    t_start = time.time()
    deadline = t_start + 55.0  # Safety margin for 60s limit
    
    MAX_N = 10_000_000
    
    # 1. Linear Sieve for Sigma (correct implementation from inspiration 2)
    lp = [0] * (MAX_N + 1)       # lowest prime factor
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
                
    if time.time() > deadline - 10:
        return [6, 11]

    # 2. Group by sigma value
    buckets = defaultdict(list)
    sig = sigma
    for i in range(1, MAX_N + 1):
        buckets[sig[i]].append(i)
        
    if time.time() > deadline - 10:
        return [6, 11]

    # 3. Find best pair using binary search within each bucket
    best_err = float('inf')
    best_pair = (6, 11)  # Fallback: sigma(6)=12, sigma(11)=12
    
    for s, vals in buckets.items():
        if len(vals) < 2:
            continue
            
        vals.sort()
        
        for b in vals:
            if time.time() > deadline - 3:
                break
                
            # We want a/b ≈ 3/2, so a ≈ 1.5*b
            target = 1.5 * b
            idx = bisect.bisect_left(vals, target)
            
            # Check idx and idx-1 (and maybe neighbors for better coverage)
            candidates = []
            for offset in [-2, -1, 0, 1, 2]:
                j = idx + offset
                if 0 <= j < len(vals):
                    candidates.append(vals[j])
            
            for a in candidates:
                if a == b:
                    continue
                if a < 1 or a > MAX_N:
                    continue
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue  # Error is 0, invalid
                err = diff / (2.0 * b)
                if err < best_err:
                    best_err = err
                    best_pair = (a, b)
                    
        if time.time() > deadline - 3:
            break

    return [int(best_pair[0]), int(best_pair[1])]
