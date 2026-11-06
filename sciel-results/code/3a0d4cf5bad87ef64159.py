import numpy as np
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin for 90s limit
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    weights = [L // i for i in range(1, n + 1)]
    
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    def gen_sums_idx(weights):
        # sums is array of int64, idxs is array of int64 (base-3 encoded)
        sums = np.array([0], dtype=np.int64)
        idxs = np.array([0], dtype=np.int64)
        
        for i, w in enumerate(weights):
            # We need to create 3 copies: -w, 0, +w
            # And update indices
            # idx digit i: 0 -> -1, 1 -> 0, 2 -> 1
            # new_idx = old_idx * 3 + digit
            
            # To save memory, we do this step by step
            # But creating 3 arrays of size 3^i is expensive if done naively.
            # Vectorized approach:
            # new_sums = np.concatenate([sums - w, sums, sums + w])
            # new_idxs = np.concatenate([idxs*3, idxs*3+1, idxs*3+2])
            
            # This creates temporary arrays.
            # Size grows by factor 3.
            # Max size ~ 14M.
            # 14M * 8 bytes = 112 MB.
            # 3 arrays = 336 MB per step.
            # This is manageable.
            
            if time.time() > deadline:
                break
                
            s_minus = sums - w
            s_zero = sums
            s_plus = sums + w
            
            i_minus = idxs * 3
            i_zero = idxs * 3 + 1
            i_plus = idxs * 3 + 2
            
            sums = np.concatenate([s_minus, s_zero, s_plus])
            idxs = np.concatenate([i_minus, i_zero, i_plus])
            
            # Free temporaries
            del s_minus, s_zero, s_plus
            del i_minus, i_zero, i_plus
            
        return sums, idxs

    sums1, idxs1 = gen_sums_idx(w1)
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]

    sums2, idxs2 = gen_sums_idx(w2)
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]

    # We want min |s1 + s2| != 0
    # Let t2 = -s2. We want min |s1 - t2| != 0
    t2 = -sums2
    del sums2
    
    # Sort sums1 and idxs1
    order1 = np.argsort(sums1)
    sorted_sums1 = sums1[order1]
    sorted_idxs1 = idxs1[order1]
    del sums1, idxs1
    
    # Sort t2 and idxs2
    order2 = np.argsort(t2)
    sorted_t2 = t2[order2]
    sorted_idxs2 = idxs2[order2]
    del t2, idxs2
    
    # Two pointer scan
    i = 0
    j = 0
    len1 = len(sorted_sums1)
    len2 = len(sorted_t2)
    
    best_diff = np.iinfo(np.int64).max
    best_i1 = -1
    best_j2 = -1
    
    while i < len1 and j < len2:
        if time.time() > deadline:
            break
            
        a = sorted_sums1[i]
        b = sorted_t2[j]
        
        if a == b:
            # Check neighbors
            # Candidate 1: i+1, j
            if i + 1 < len1:
                diff = abs(sorted_sums1[i+1] - b)
                if diff < best_diff:
                    best_diff = diff
                    best_i1 = i + 1
                    best_j2 = j
            # Candidate 2: i, j+1
            if j + 1 < len2:
                diff = abs(a - sorted_t2[j+1])
                if diff < best_diff:
                    best_diff = diff
                    best_i1 = i
                    best_j2 = j + 1
            
            # Move both pointers?
            # If a == b, then a is not a candidate.
            # Any pair (a, b') with b' > b has diff >= b' - b.
            # Any pair (a', b) with a' > a has diff >= a' - a.
            # So we can increment both.
            i += 1
            j += 1
        elif a < b:
            diff = b - a
            if diff < best_diff:
                best_diff = diff
                best_i1 = i
                best_j2 = j
            i += 1
        else:
            diff = a - b
            if diff < best_diff:
                best_diff = diff
                best_i1 = i
                best_j2 = j
            j += 1

    if best_i1 == -1:
        # Fallback: just return a valid non-zero sum
        # e.g. c_1 = 1, others 0. Sum = 1.
        return [1] + [0]*29

    idx1 = sorted_idxs1[best_i1]
    idx2 = sorted_idxs2[best_j2]
    
    def decode(idx, k):
        coeffs = []
        rem = idx
        for _ in range(k):
            digit = rem % 3
            rem //= 3
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
        return coeffs

    c1 = decode(idx1, half)
    c2 = decode(idx2, n - half)
    
    return c1 + c2
