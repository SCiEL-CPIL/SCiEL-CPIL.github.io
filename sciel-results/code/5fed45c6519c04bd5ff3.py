import numpy as np
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 second safety margin out of 60s
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    weights = [L // i for i in range(1, n + 1)]
    
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Generate sums and indices for a given weight list
    # Uses recursive doubling which tracks base-3 indices
    def gen_sums(w):
        sums = [0]
        idxs = [0]
        for i, weight in enumerate(w):
            pow3 = 3 ** i
            new_sums = []
            new_idxs = []
            for s, idx in zip(sums, idxs):
                new_sums.append(s - weight)
                new_idxs.append(idx)
                
                new_sums.append(s)
                new_idxs.append(idx + pow3)
                
                new_sums.append(s + weight)
                new_idxs.append(idx + 2 * pow3)
            sums = new_sums
            idxs = new_idxs
        return sums, idxs
    
    # Generate first half sums with indices
    sums1, idxs1 = gen_sums(w1)
    
    # Prepare for binary search using NumPy
    sums1_arr = np.array(sums1, dtype=np.int64)
    idxs1_arr = np.array(idxs1, dtype=np.int64)
    
    sorted_order = np.argsort(sums1_arr)
    sorted_sums1_vals = sums1_arr[sorted_order]
    sorted_idxs1 = idxs1_arr[sorted_order]
    
    best_abs = float('inf')
    best_idx1 = -1
    best_idx2 = -1
    
    # Generate second half sums with indices
    sums2, idxs2 = gen_sums(w2)
    
    # Iterate over second half sums
    for s2, idx2 in zip(sums2, idxs2):
        if time.time() > deadline:
            break
            
        target = -s2
        # Find closest in sorted_sums1_vals
        pos = np.searchsorted(sorted_sums1_vals, target)
        
        # Check pos and pos-1
        if pos < len(sorted_sums1_vals):
            s1_val = sorted_sums1_vals[pos]
            total_sum = s1_val + s2
            if total_sum != 0:
                abs_sum = abs(total_sum)
                if abs_sum < best_abs:
                    best_abs = abs_sum
                    best_idx1 = sorted_idxs1[pos]
                    best_idx2 = idx2
                    if best_abs == 1:
                        break
        
        if pos > 0:
            s1_val = sorted_sums1_vals[pos - 1]
            total_sum = s1_val + s2
            if total_sum != 0:
                abs_sum = abs(total_sum)
                if abs_sum < best_abs:
                    best_abs = abs_sum
                    best_idx1 = sorted_idxs1[pos - 1]
                    best_idx2 = idx2
                    if best_abs == 1:
                        break

    if best_idx1 == -1:
        # Fallback: return a valid non-zero sum
        return [1] + [0]*29

    # Decode indices to coefficients
    def decode(idx, k):
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

    c1 = decode(best_idx1, half)
    c2 = decode(best_idx2, n - half)
    
    result = c1 + c2
    
    # Ensure types are standard Python ints
    return [int(x) for x in result]
