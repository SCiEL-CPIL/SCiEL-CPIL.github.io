import time
import numpy as np
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n in [1, N] using NumPy sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline - 20:
        return [11, 6]

    # 2. Group numbers by sigma value
    # We use a dictionary mapping sigma value to a list of n's
    groups = {}
    
    # To speed up iteration, we can iterate n and append to groups
    # sigma is a numpy array, so sigma[n] is a numpy scalar, need to convert to int for dict key
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s not in groups:
            groups[s] = []
        groups[s].append(n)
        
    if time.time() > deadline - 10:
        return [11, 6]

    # 3. Sort each group to allow binary search
    for s, lst in groups.items():
        # lst is already sorted because we iterated n from 1 to N
        # But just in case, or to ensure type is list of ints
        pass # Already sorted

    # 4. Search for best pair
    best_a = 11
    best_b = 6
    best_err = 1.0 / 3.0  # |11/6 - 1.5| = 1/3
    
    # We want to minimize E = |2a - 3b| / (2b)
    # E < best_err  =>  |2a - 3b| < 2 * b * best_err
    # Since |2a - 3b| >= 1 (integer), we need 1 < 2 * b * best_err
    # b > 1 / (2 * best_err)
    
    threshold_b = 1.0 / (2.0 * best_err)
    
    # Iterate b from N down to 1
    # We can break if b <= threshold_b
    
    # To avoid overhead of checking time in the inner loop, we check every 10000 steps
    for b in range(N, 1, -1):
        if b <= threshold_b:
            break
            
        if b % 10000 == 0 and time.time() > deadline - 5:
            break
            
        s = int(sigma[b])
        lst = groups.get(s)
        if not lst or len(lst) < 2:
            continue
            
        target = 1.5 * b
        # Find position in lst
        idx = bisect_left(lst, target)
        
        # Check idx and idx - 1
        candidates = []
        if idx < len(lst):
            candidates.append(lst[idx])
        if idx > 0:
            candidates.append(lst[idx - 1])
            
        for a in candidates:
            if a == b:
                continue
            # Calculate error
            # E = |2a - 3b| / (2b)
            # We can compare |2a - 3b| / b < 2 * best_err
            # Or just calculate float
            diff = abs(2 * a - 3 * b)
            # If diff is 0, a/b = 1.5 exactly, error 0. But a,b integers, 2a=3b => a=1.5b.
            # If b even, a integer. e.g. b=2, a=3. sigma(3)=4, sigma(2)=3. Not equal.
            # If such a pair exists, error is 0.
            err = diff / (2.0 * b)
            if err < best_err:
                best_err = err
                best_a = a
                best_b = b
                threshold_b = 1.0 / (2.0 * best_err)
                
    return [int(best_a), int(best_b)]
