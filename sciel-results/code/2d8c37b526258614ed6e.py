import time
import math
from collections import defaultdict
import bisect
from array import array

def construct():
    t_start = time.time()
    deadline = t_start + 55.0  # Safety margin for 90s limit
    
    MAX_N = 10_000_000
    
    # 1. Linear Sieve for Sigma using arrays to reduce memory
    lp = array('I', [0]) * (MAX_N + 1)       # lowest prime (uint32)
    pow_lp = array('I', [0]) * (MAX_N + 1)   # highest power of lp dividing n (uint32)
    h = array('Q', [0]) * (MAX_N + 1)        # sum of powers of lp (uint64)
    sigma = array('Q', [0]) * (MAX_N + 1)    # sigma(n) (uint64)
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
            
        li = lp[i]
        pi_ = i
        for p in primes:
            if p > li or pi_ * p > MAX_N:
                break
            ip = pi_ * p
            lp[ip] = p
            
            if p == li:
                # p divides i
                m = i // pow_lp[i]
                pow_lp[ip] = pow_lp[i] * p
                h[ip] = h[i] + pow_lp[i] * p
                sigma[ip] = sigma[m] * h[ip]
                break
            else:
                # p < lp[i], so gcd(i, p) = 1
                pow_lp[ip] = p
                h[ip] = p + 1
                sigma[ip] = sigma[i] * (p + 1)
                
    if time.time() > deadline - 5:
        return [6, 11]

    # 2. Group by sigma value
    buckets = defaultdict(list)
    sig = sigma
    for i in range(1, MAX_N + 1):
        buckets[sig[i]].append(i)
        
    if time.time() > deadline - 5:
        return [6, 11]

    # 3. Find best pair
    best_err = float('inf')
    best_pair = (6, 11) # Fallback
    
    for s, vals in buckets.items():
        if len(vals) < 2:
            continue
            
        vals.sort()
        
        for b in vals:
            if time.time() > deadline - 3:
                break
                
            target = 1.5 * b
            idx = bisect.bisect_left(vals, target)
            
            if idx < len(vals):
                a = vals[idx]
                if a != b:
                    diff = abs(2 * a - 3 * b)
                    if diff != 0:
                        err = diff / (2.0 * b)
                        if err < best_err:
                            best_err = err
                            best_pair = (a, b)
            if idx > 0:
                a = vals[idx-1]
                if a != b:
                    diff = abs(2 * a - 3 * b)
                    if diff != 0:
                        err = diff / (2.0 * b)
                        if err < best_err:
                            best_err = err
                            best_pair = (a, b)
    
    # 4. Secondary targeted search: for each b, check a near 1.5*b
    #    using on-the-fly sigma via the precomputed lp array
    if time.time() < deadline - 8:
        # Build a fast sigma lookup using the lp/pow_lp/h arrays
        # Actually we already have sigma[] array, so just index into it
        sig_arr = sigma
        
        # Iterate b from larger values down (larger b -> smaller potential error)
        # Focus on b where 1.5*b is close to an integer, i.e., b even
        # a = round(1.5*b), check a and nearby values
        step = 1
        for b in range(MAX_N, 0, -step):
            if time.time() > deadline - 3:
                break
            
            sb = sig_arr[b]
            if sb == 0:
                continue
            
            # a should be close to 1.5*b
            a_center = (3 * b) // 2
            # Check a in a small window around a_center
            # The error is |2a - 3b| / (2b), so we want 2a close to 3b
            # 2a = 3b + d, error = |d|/(2b)
            # For even b: 3b is odd, so 2a (even) can't equal 3b (odd) -> min |d| = 1
            # For odd b: 3b is odd, 2a even -> min |d| = 1 as well
            # So minimum possible error for any pair is 1/(2b)
            
            # Check a = a_center and a_center+1 (to cover both parities)
            for a in (a_center, a_center + 1):
                if a < 1 or a > MAX_N:
                    continue
                if a == b:
                    continue
                if sig_arr[a] == sb:
                    diff = abs(2 * a - 3 * b)
                    if diff == 0:
                        continue
                    err = diff / (2.0 * b)
                    if err < best_err:
                        best_err = err
                        best_pair = (a, b)
    
    return [int(best_pair[0]), int(best_pair[1])]
