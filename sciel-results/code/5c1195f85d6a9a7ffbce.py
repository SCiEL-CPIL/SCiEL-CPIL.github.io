import time
import numpy as np
from math import lcm
from bisect import bisect_left, bisect_right

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    n = 30
    # LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Integer weights: w_i = L // (i+1) for coefficient c_i multiplying 1/(i+1)
    # Sum = sum(c_i * L/(i+1)) / L
    # Minimize |sum(c_i * w_i)| where w_i = L//(i+1)
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    # Split into two halves of 15
    half1 = weights[:15]
    half2 = weights[15:]
    
    # Generate all possible integer sums for each half with their coefficient vectors
    # Use iterative approach with numpy for speed
    # Store (sum, coeff_vector) pairs to enable recovery
    
    def gen_sums_with_coeffs(w):
        """Generate all 3^len(w) possible sums with coefficient vectors using numpy"""
        # Start with sum=0, coeffs=[0]
        sums = np.array([0], dtype=np.int64)
        coeffs_list = [np.zeros(len(w), dtype=np.int8)]
        
        for wi_idx, wi in enumerate(w):
            # For each existing sum, create sum-wi, sum, sum+wi
            new_sums = np.concatenate([sums - wi, sums, sums + wi])
            
            new_coeffs_list = []
            for c in coeffs_list:
                # c with -1 at position wi_idx
                c_neg = c.copy()
                c_neg[wi_idx] = -1
                new_coeffs_list.append(c_neg)
                # c unchanged (0 at position wi_idx)
                c_zero = c.copy()
                c_zero[wi_idx] = 0
                new_coeffs_list.append(c_zero)
                # c with +1 at position wi_idx
                c_pos = c.copy()
                c_pos[wi_idx] = 1
                new_coeffs_list.append(c_pos)
            
            sums = new_sums
            coeffs_list = new_coeffs_list
        
        return sums, coeffs_list
    
    # Generate sums for both halves
    sums1, coeffs1_list = gen_sums_with_coeffs(half1)
    sums2, coeffs2_list = gen_sums_with_coeffs(half2)
    
    # Create sorted index for sums2
    sums2_sorted, idx2 = np.sort(sums2), np.argsort(sums2)
    sums2_sorted_list = sums2_sorted.tolist()
    
    # For each sum in sums1, find the closest sum in sums2 such that s1+s2 != 0
    min_abs = np.iinfo(np.int64).max
    best_s1_idx = -1
    best_s2_idx = -1
    
    n1 = len(sums1)
    
    for i in range(n1):
        if time.time() > deadline:
            break
            
        s1 = sums1[i]
        
        # We want s1 + s2 close to 0, so s2 close to -s1
        target = -s1
        
        # Find insertion point of target in sums2_sorted
        pos = bisect_left(sums2_sorted_list, target)
        
        # Check positions pos-1 and pos (and nearby) for closest nonzero sum
        candidates = []
        for p in range(max(0, pos-2), min(len(sums2_sorted_list), pos+3)):
            s2 = sums2_sorted_list[p]
            total = s1 + s2
            if total != 0:
                candidates.append((abs(total), p))
        
        if candidates:
            candidates.sort()
            abs_val, p = candidates[0]
            if abs_val < min_abs:
                min_abs = abs_val
                best_s1_idx = i
                best_s2_idx = p
                if min_abs == 1:
                    break
    
    if best_s1_idx == -1 or min_abs == np.iinfo(np.int64).max:
        # Fallback: return a simple nonzero solution
        fallback = [0] * 28 + [1, -1]
        return fallback
    
    # Recover coefficients
    c1 = coeffs1_list[best_s1_idx].tolist()
    c2 = coeffs2_list[idx2[best_s2_idx]].tolist()
    
    result = c1 + c2
    
    # Verify the result
    total_int = sum(c * w for c, w in zip(result, weights))
    if total_int == 0:
        # Should not happen, but fallback
        fallback = [0] * 28 + [1, -1]
        return fallback
        
    return result
