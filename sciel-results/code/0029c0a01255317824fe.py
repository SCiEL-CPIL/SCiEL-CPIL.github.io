import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # Weights for i=1..30 are L/i
    # Indices 0..29 correspond to denominators 1..30
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums_and_codes(weights_half):
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int64)
        
        for w in weights_half:
            # New sums
            s_minus = sums - w
            s_zero = sums
            s_plus = sums + w
            
            # New codes
            # Digit 0 -> -1, Digit 1 -> 0, Digit 2 -> 1
            # We append the new digit, so it becomes the least significant digit of the new code
            # Wait, if we do code * 3 + digit, the old code shifts left (becomes more significant).
            # So the first weight processed becomes the most significant.
            c_minus = codes * 3      # digit 0
            c_zero = codes * 3 + 1   # digit 1
            c_plus = codes * 3 + 2   # digit 2
            
            sums = np.concatenate([s_minus, s_zero, s_plus])
            codes = np.concatenate([c_minus, c_zero, c_plus])
            
        return sums, codes

    # Generate for first half
    sums1, codes1 = generate_sums_and_codes(w1)
    
    # Sort first half sums and align codes
    order1 = np.argsort(sums1)
    sums1_sorted = sums1[order1]
    codes1_sorted = codes1[order1]
    
    # Generate for second half
    sums2, codes2 = generate_sums_and_codes(w2)
    
    # Find best pair
    # For each s2 in sums2, we want s1 such that s1 + s2 is close to 0.
    # Target s1 = -s2.
    
    len_s1 = len(sums1_sorted)
    
    # Vectorized search
    # searchsorted returns indices
    # We check predecessor (idx-1) and successor (idx)
    
    targets = -sums2
    
    # Predecessor
    idx_l = np.searchsorted(sums1_sorted, targets, side='left')
    # idx_l is the first index >= target.
    # So idx_l - 1 is the largest < target (if target exists, idx_l points to it, so idx_l-1 is < target? No, if target exists, idx_l points to first occurrence. If we want closest, we should check idx_l and idx_l-1).
    # Actually, if target is in array, idx_l is first == target.
    # If target not in array, idx_l is first > target.
    # In both cases, idx_l - 1 is the largest < target (or <= if we consider the value at idx_l might be equal).
    # Let's just check idx_l - 1 and idx_l.
    
    # Handle bounds
    idx_pred = idx_l - 1
    valid_pred = idx_pred >= 0
    idx_pred_clipped = np.clip(idx_pred, 0, len_s1 - 1)
    
    s1_pred = sums1_sorted[idx_pred_clipped]
    c1_pred = codes1_sorted[idx_pred_clipped]
    
    total_pred = sums2 + s1_pred
    abs_total_pred = np.abs(total_pred)
    
    # Mask invalid
    abs_total_pred[~valid_pred] = np.iinfo(np.int64).max
    
    # Successor
    idx_succ = idx_l
    valid_succ = idx_succ < len_s1
    idx_succ_clipped = np.clip(idx_succ, 0, len_s1 - 1)
    
    s1_succ = sums1_sorted[idx_succ_clipped]
    c1_succ = codes1_sorted[idx_succ_clipped]
    
    total_succ = sums2 + s1_succ
    abs_total_succ = np.abs(total_succ)
    
    # Mask invalid
    abs_total_succ[~valid_succ] = np.iinfo(np.int64).max
    
    # Find global minimum
    min_pred = np.min(abs_total_pred)
    min_succ = np.min(abs_total_succ)
    
    if min_pred < min_succ:
        best_abs = min_pred
        idx_best_2 = np.argmin(abs_total_pred)
        best_s1_idx = idx_pred_clipped[idx_best_2]
        best_code1 = c1_pred[idx_best_2]
        best_code2 = codes2[idx_best_2]
        # Verify it's not zero
        if best_abs == 0:
            # This shouldn't happen if we exclude 0, but if it does, we need to find next best.
            # However, the problem says "nonzero".
            # If min is 0, we must exclude it.
            # Let's handle this by masking 0s before argmin?
            # It's easier to just check and if 0, re-run or find next.
            # Given the density, 0 is likely possible?
            # Sum of 1/i is irrational? No, rational.
            # Is 0 achievable?
            # If 0 is achievable, we must find the smallest non-zero.
            # Let's modify the masks to exclude 0.
            pass
    
    else:
        best_abs = min_succ
        idx_best_2 = np.argmin(abs_total_succ)
        best_s1_idx = idx_succ_clipped[idx_best_2]
        best_code1 = c1_succ[idx_best_2]
        best_code2 = codes2[idx_best_2]

    # Robust handling of Zero Sum
    # If best_abs is 0, we need the next smallest.
    # We can just filter out 0s from the candidate arrays and re-min.
    # To be safe and simple, let's do a second pass if best_abs == 0.
    
    if best_abs == 0:
        # Mask out zeros in both pred and succ candidates
        mask_pred = valid_pred & (total_pred != 0)
        abs_total_pred_safe = abs_total_pred.copy()
        abs_total_pred_safe[~mask_pred] = np.iinfo(np.int64).max
        
        mask_succ = valid_succ & (total_succ != 0)
        abs_total_succ_safe = abs_total_succ.copy()
        abs_total_succ_safe[~mask_succ] = np.iinfo(np.int64).max
        
        min_pred_safe = np.min(abs_total_pred_safe)
        min_succ_safe = np.min(abs_total_succ_safe)
        
        if min_pred_safe < min_succ_safe:
            best_abs = min_pred_safe
            idx_best_2 = np.argmin(abs_total_pred_safe)
            best_code1 = c1_pred[idx_best_2]
            best_code2 = codes2[idx_best_2]
        else:
            best_abs = min_succ_safe
            idx_best_2 = np.argmin(abs_total_succ_safe)
            best_code1 = c1_succ[idx_best_2]
            best_code2 = codes2[idx_best_2]

    # Decode codes
    def decode_code(code, num_weights):
        coeffs = [0] * num_weights
        c = int(code)
        for i in range(num_weights - 1, -1, -1):
            digit = c % 3
            c //= 3
            # 0 -> -1, 1 -> 0, 2 -> 1
            coeffs[i] = digit - 1
        return coeffs

    c1 = decode_code(best_code1, len(w1))
    c2 = decode_code(best_code2, len(w2))
    
    final_coeffs = c1 + c2
    
    # Final verification (optional, but good for debugging)
    # val = sum(final_coeffs[i] / (i+1) for i in range(n))
    # assert val != 0
    
    return final_coeffs
