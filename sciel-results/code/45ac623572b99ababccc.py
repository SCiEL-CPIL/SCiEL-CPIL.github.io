import time
from math import gcd
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin
    
    n = 30
    L = reduce(lambda a, b: a * b // gcd(a, b), range(1, n + 1), 1)
    weights = [L // i for i in range(1, n + 1)]
    
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    size1 = 3 ** mid
    size2 = 3 ** (n - mid)
    
    # Generate sums for half 1
    sums1 = []
    for idx in range(size1):
        s = 0
        t = idx
        for i in range(mid):
            digit = t % 3
            t //= 3
            if digit == 0:
                s -= w1[i]
            elif digit == 2:
                s += w1[i]
        sums1.append((s, idx))
        
    if time.time() - start_time > timeout:
        return [0] * 28 + [1, -1]
        
    sums1.sort(key=lambda x: x[0])
    sorted_sums1 = [s for s, idx in sums1]
    
    best_diff = None
    best_idx1 = 0
    best_idx2 = 0
    
    for idx2 in range(size2):
        if time.time() - start_time > timeout:
            break
            
        s2 = 0
        t = idx2
        for i in range(n - mid):
            digit = t % 3
            t //= 3
            if digit == 0:
                s2 -= w2[i]
            elif digit == 2:
                s2 += w2[i]
                
        target = -s2
        pos = bisect.bisect_left(sorted_sums1, target)
        
        candidates = []
        if pos < size1:
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
            
        for p in candidates:
            s1 = sums1[p][0]
            idx1 = sums1[p][1]
            diff = s1 + s2
            if diff == 0:
                continue
            if best_diff is None or abs(diff) < abs(best_diff):
                best_diff = diff
                best_idx1 = idx1
                best_idx2 = idx2
                
    if best_diff is None:
        return [0] * 28 + [1, -1]
        
    coeffs = [0] * n
    
    t = best_idx1
    for i in range(mid):
        digit = t % 3
        t //= 3
        if digit == 0:
            coeffs[i] = -1
        elif digit == 2:
            coeffs[i] = 1
        else:
            coeffs[i] = 0
            
    t = best_idx2
    for i in range(n - mid):
        digit = t % 3
        t //= 3
        if digit == 0:
            coeffs[mid + i] = -1
        elif digit == 2:
            coeffs[mid + i] = 1
        else:
            coeffs[mid + i] = 0
            
    return coeffs
