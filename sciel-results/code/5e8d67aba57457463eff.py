import numpy as np
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5 second safety margin
    
    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Weights
    weights = [L // i for i in range(1, n + 1)]
    
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Generate sums for first half
    # 3^15 = 14348907
    # We need to be efficient.
    
    def gen_sums(w):
        sums = []
        k = len(w)
        total = 3 ** k
        for idx in range(total):
            s = 0
            rem = idx
            for j in range(k):
                digit = rem % 3
                rem //= 3
                if digit == 0:
                    c = -1
                elif digit == 1:
                    c = 0
                else:
                    c = 1
                s += c * w[j]
            sums.append((s, idx))
        sums.sort(key=lambda x: x[0])
        return sums

    # Check time
    if time.time() > deadline - 10:
         # Fallback to simple heuristic if generation is too slow (unlikely)
         pass

    sums1 = gen_sums(w1)
    
    # Extract sorted sums for binary search
    sorted_sums1_vals = np.array([s[0] for s in sums1])
    
    best_abs = float('inf')
    best_idx1 = -1
    best_idx2 = -1
    
    k2 = len(w2)
    total2 = 3 ** k2
    
    for idx2 in range(total2):
        s2 = 0
        rem = idx2
        for j in range(k2):
            digit = rem % 3
            rem //= 3
            if digit == 0:
                c = -1
            elif digit == 1:
                c = 0
            else:
                c = 1
            s2 += c * w2[j]
            
        target = -s2
        # Find closest in sorted_sums1_vals
        pos = np.searchsorted(sorted_sums1_vals, target)
        
        # Check pos and pos-1
        candidates = []
        if pos < len(sorted_sums1_vals):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
            
        for p in candidates:
            s1_val, idx1 = sums1[p]
            total_sum = s1_val + s2
            if total_sum == 0:
                continue
            abs_sum = abs(total_sum)
            if abs_sum < best_abs:
                best_abs = abs_sum
                best_idx1 = idx1
                best_idx2 = idx2
                
        if time.time() > deadline:
            break

    if best_idx1 == -1:
        # Fallback
        return [0]*28 + [1, -1]

    # Decode indices to coefficients
    def decode(idx, k, w):
        coeffs = []
        rem = idx
        for j in range(k):
            digit = rem % 3
            rem //= 3
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
        return coeffs

    c1 = decode(best_idx1, half, w1)
    c2 = decode(best_idx2, n - half, w2)
    
    return c1 + c2
