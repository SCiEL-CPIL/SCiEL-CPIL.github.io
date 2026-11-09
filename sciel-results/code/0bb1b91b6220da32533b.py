import time
import numpy as np
from math import gcd

def _lcm_range(n):
    result = 1
    for i in range(2, n + 1):
        result = result * i // gcd(result, i)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 30
    lcm_val = _lcm_range(n)
    
    # Weights for integer arithmetic: S = sum(c_i / i) = (1/LCM) * sum(c_i * LCM/i)
    # c_i in {-1, 0, 1}
    weights = [lcm_val // i for i in range(1, n + 1)]
    
    # Split into two halves for meet-in-the-middle
    half = 15
    w1 = np.array(weights[:half], dtype=np.int64)
    w2 = np.array(weights[half:], dtype=np.int64)
    
    # Generate all possible sums for the first half
    # There are 3^15 = 14,348,907 combinations.
    def gen_all_sums(w_part):
        n_part = len(w_part)
        sums = np.array([0], dtype=np.int64)
        for i in range(n_part):
            a_i = w_part[i]
            # Efficiently create new array by concatenating
            # sums + a_i, sums, sums - a_i
            # Order: +1, 0, -1
            sums = np.concatenate([sums + a_i, sums, sums - a_i])
        return sums

    # Check time
    if time.time() > deadline - 10:
        return [0] * 28 + [1, -1]

    sums1 = gen_all_sums(w1)
    
    if time.time() > deadline - 10:
        return [0] * 28 + [1, -1]

    sums2 = gen_all_sums(w2)
    
    # Sort sums1 and keep track of indices
    n1 = len(sums1)
    order1 = np.argsort(sums1)
    sorted_sums1 = sums1[order1]
    
    # sums2 is unsorted, indices are 0..n2-1
    n2 = len(sums2)
    
    best_abs = np.iinfo(np.int64).max
    best_idx1 = -1
    best_idx2 = -1
    
    # Process in chunks to manage memory
    chunk_size = 1000000
    
    for chunk_start in range(0, n2, chunk_size):
        if time.time() > deadline - 5:
            break
        
        chunk_end = min(chunk_start + chunk_size, n2)
        s2_chunk = sums2[chunk_start:chunk_end]
        targets = -s2_chunk
        
        # Find insertion points in sorted sums1
        pos = np.searchsorted(sorted_sums1, targets)
        
        # Candidate 1: sums1[pos] (clipped)
        pos1 = np.clip(pos, 0, n1 - 1)
        s1_cand1 = sorted_sums1[pos1]
        total1 = s1_cand1 + s2_chunk
        abs_total1 = np.abs(total1)
        
        # Candidate 2: sums1[pos-1] (clipped)
        pos2 = np.clip(pos - 1, 0, n1 - 1)
        s1_cand2 = sorted_sums1[pos2]
        total2 = s1_cand2 + s2_chunk
        abs_total2 = np.abs(total2)
        
        # Mask out zeros
        abs_total1[total1 == 0] = np.iinfo(np.int64).max
        abs_total2[total2 == 0] = np.iinfo(np.int64).max
        
        # Find min in this chunk
        min1 = np.min(abs_total1)
        min2 = np.min(abs_total2)
        
        if min1 < best_abs:
            idx_in_chunk = np.argmin(abs_total1)
            best_abs = min1
            # Get original index in sums1
            # pos1[idx_in_chunk] is the index in sorted_sums1
            # order1[...] gives the index in original sums1
            best_idx1 = int(order1[pos1[idx_in_chunk]])
            best_idx2 = int(chunk_start + idx_in_chunk)
            
            if best_abs == 1:
                break
        
        if min2 < best_abs:
            idx_in_chunk = np.argmin(abs_total2)
            best_abs = min2
            best_idx1 = int(order1[pos2[idx_in_chunk]])
            best_idx2 = int(chunk_start + idx_in_chunk)
            
            if best_abs == 1:
                break

    if best_idx1 == -1:
        # Fallback: simple nonzero sum
        return [0] * 28 + [1, -1]
    
    # Reconstruct coefficients
    def decode_coeffs(idx, w_part):
        n_part = len(w_part)
        coeffs = [0] * n_part
        # Mapping: digit 0 -> 1, digit 1 -> 0, digit 2 -> -1
        # idx is base-3 representation
        # c_j corresponds to digit j (0 is LSB)
        # w_part[j] is the weight for c_j
        for j in range(n_part):
            digit = (idx // (3 ** j)) % 3
            if digit == 0:
                coeffs[j] = 1
            elif digit == 1:
                coeffs[j] = 0
            else:
                coeffs[j] = -1
        return coeffs

    c1 = decode_coeffs(best_idx1, w1)
    c2 = decode_coeffs(best_idx2, w2)
    
    result = [0] * 30
    for i in range(half):
        result[i] = int(c1[i])
    for i in range(half):
        result[half + i] = int(c2[i])
    
    return result
