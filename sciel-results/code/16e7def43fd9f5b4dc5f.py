"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from math import gcd

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    N = 30
    half = 15
    
    # Denominators are 1, 2, ..., 30
    # Use global LCM as common denominator
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Compute LCM of 1..30
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
    # L = 465817912560
    
    # For each denominator d, the coefficient c/d = c * (L/d) / L
    # So we need integer weights w[i] = L // (i+1) for i=0..29
    weights = np.array([L // (i + 1) for i in range(N)], dtype=np.int64)
    
    # Split into two halves
    w1 = weights[:half]  # denominators 1..15
    w2 = weights[half:]  # denominators 16..30
    
    def gen_sums_with_codes(weights_arr):
        """Generate all possible sums AND their base-3 codes for given weights."""
        n = len(weights_arr)
        # Start with sum=0, code=0
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int64)
        
        for i, w in enumerate(weights_arr):
            # For each existing (sum, code), create three new entries
            # code_digit 0 -> c=-1, code_digit 1 -> c=0, code_digit 2 -> c=+1
            # New code = old_code * 3 + digit
            # New sum = old_sum + c * w
            
            new_sums = np.concatenate([sums - w, sums, sums + w])
            new_codes = np.concatenate([
                codes * 3 + 0,  # c = -1
                codes * 3 + 1,  # c = 0
                codes * 3 + 2   # c = +1
            ])
            sums = new_sums
            codes = new_codes
        
        return sums, codes
    
    def decode_code(code, n):
        """Decode a base-3 code into coefficients in {-1, 0, 1}."""
        coeffs = []
        for i in range(n):
            digit = code % 3
            code //= 3
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
        coeffs.reverse()
        return coeffs
    
    # Generate sums and codes for half 1
    sums1, codes1 = gen_sums_with_codes(w1)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20.0:
        return [0]*28 + [1, -1]
    
    # Generate sums and codes for half 2
    sums2, codes2 = gen_sums_with_codes(w2)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10.0:
        return [0]*28 + [1, -1]
    
    # Sort sums1 and keep track of original indices
    order1 = np.argsort(sums1)
    sums1_sorted = sums1[order1]
    codes1_sorted = codes1[order1]
    
    # For each value in sums2, find the value in sums1 that makes sum closest to 0
    neg_sums2 = -sums2
    idx = np.searchsorted(sums1_sorted, neg_sums2)
    
    best_abs = None
    best_code1 = None
    best_code2 = None
    
    # Process in chunks to manage memory
    chunk_size = 200000
    for start in range(0, len(sums2), chunk_size):
        end = min(start + chunk_size, len(sums2))
        idx_chunk = idx[start:end]
        
        # Candidate indices: idx-1 and idx
        idx_left = np.clip(idx_chunk - 1, 0, len(sums1_sorted) - 1)
        idx_right = np.clip(idx_chunk, 0, len(sums1_sorted) - 1)
        
        a_left = sums1_sorted[idx_left]
        a_right = sums1_sorted[idx_right]
        
        b_chunk = sums2[start:end]
        codes2_chunk = codes2[start:end]
        
        # Compute sums
        sum_left = a_left + b_chunk
        sum_right = a_right + b_chunk
        
        abs_left = np.abs(sum_left)
        abs_right = np.abs(sum_right)
        
        # Exclude zero sums
        max_int = np.iinfo(np.int64).max
        abs_left[sum_left == 0] = max_int
        abs_right[sum_right == 0] = max_int
        
        # Find best in this chunk
        min_left = np.min(abs_left)
        min_right = np.min(abs_right)
        
        if min_left < min_right:
            local_best_idx = np.argmin(abs_left)
            local_best_abs = min_left
            local_best_code1 = codes1_sorted[idx_left[local_best_idx]]
            local_best_code2 = codes2_chunk[local_best_idx]
        else:
            local_best_idx = np.argmin(abs_right)
            local_best_abs = min_right
            local_best_code1 = codes1_sorted[idx_right[local_best_idx]]
            local_best_code2 = codes2_chunk[local_best_idx]
        
        if best_abs is None or local_best_abs < best_abs:
            best_abs = local_best_abs
            best_code1 = local_best_code1
            best_code2 = local_best_code2
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
    
    if best_abs is None or best_abs == np.iinfo(np.int64).max:
        return [0]*28 + [1, -1]
    
    # Decode the codes to get coefficients
    coeffs1 = decode_code(int(best_code1), half)
    coeffs2 = decode_code(int(best_code2), half)
    
    # Combine and convert to Python ints
    result = [int(c) for c in coeffs1] + [int(c) for c in coeffs2]
    
    # Verify: compute exact sum using fractions
    from fractions import Fraction
    total = Fraction(0, 1)
    for i in range(N):
        total += Fraction(result[i], i + 1)
    
    if total == 0:
        return [0]*28 + [1, -1]
    
    return result

# EVOLVE-BLOCK-END
