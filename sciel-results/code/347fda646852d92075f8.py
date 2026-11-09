import time
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, 31):
        L = L * i // gcd(L, i)
    
    # A[i] = L // i, so sum c[i-1]/i = (sum c[i-1]*A[i]) / L
    A = [L // i for i in range(1, N + 1)]
    
    # Split into two halves
    half1 = list(range(0, 15))  # indices 0..14 correspond to denominators 1..15
    half2 = list(range(15, 30))  # indices 15..29 correspond to denominators 16..30
    
    A1 = np.array([A[i] for i in half1], dtype=np.int64)
    A2 = np.array([A[i] for i in half2], dtype=np.int64)
    
    # Generate all 3^15 signed sums and their packed coefficients
    def gen_all_sums_packed(A_part):
        n = len(A_part)
        sums = np.array([0], dtype=np.int64)
        coeffs = np.array([0], dtype=np.int64)
        
        for i in range(n):
            a_i = A_part[i]
            
            # Part 1: +1 (bit pattern 01)
            s1 = sums + a_i
            c1 = coeffs | (1 << (2 * i))
            
            # Part 2: 0 (bit pattern 00)
            s2 = sums
            c2 = coeffs
            
            # Part 3: -1 (bit pattern 10)
            s3 = sums - a_i
            c3 = coeffs | (2 << (2 * i))
            
            sums = np.concatenate([s1, s2, s3])
            coeffs = np.concatenate([c1, c2, c3])
            
        return sums, coeffs
    
    sums1, coeffs1 = gen_all_sums_packed(A1)
    order1 = np.argsort(sums1)
    sums1_sorted = sums1[order1]
    coeffs1_sorted = coeffs1[order1]
    
    sums2, coeffs2 = gen_all_sums_packed(A2)
    
    # For each s2, find s1 such that s1 + s2 is closest to 0 but not 0
    best_abs = None
    best_idx1 = None
    best_idx2 = None
    
    n1 = len(sums1_sorted)
    n2 = len(sums2)
    
    # Process in chunks to manage memory
    chunk_size = 200000
    
    for chunk_start in range(0, n2, chunk_size):
        if time.time() > deadline - 10:
            break
        
        chunk_end = min(chunk_start + chunk_size, n2)
        s2_chunk = sums2[chunk_start:chunk_end]
        targets = -s2_chunk
        
        # Find insertion points in sorted sums1
        pos = np.searchsorted(sums1_sorted, targets)
        
        # Candidate 1: sums1[pos] (clipped)
        pos_clipped = np.clip(pos, 0, n1 - 1)
        s1_cand1 = sums1_sorted[pos_clipped]
        total1 = s1_cand1 + s2_chunk
        abs_total1 = np.abs(total1)
        
        # Candidate 2: sums1[pos-1] (clipped)
        pos_minus = np.clip(pos - 1, 0, n1 - 1)
        s1_cand2 = sums1_sorted[pos_minus]
        total2 = s1_cand2 + s2_chunk
        abs_total2 = np.abs(total2)
        
        # Combine and find minimum nonzero
        all_abs = np.concatenate([abs_total1, abs_total2])
        all_total = np.concatenate([total1, total2])
        
        # Mask out zeros
        nonzero_mask = all_total != 0
        if not np.any(nonzero_mask):
            continue
        
        abs_nonzero = np.where(nonzero_mask, all_abs, np.iinfo(np.int64).max)
        min_idx_local = np.argmin(abs_nonzero)
        min_abs = int(abs_nonzero[min_idx_local])
        
        if best_abs is None or min_abs < best_abs:
            best_abs = min_abs
            # Determine which candidate was the min
            if min_idx_local < len(abs_total1):
                idx_in_chunk = min_idx_local
                cand_type = 1
            else:
                idx_in_chunk = min_idx_local - len(abs_total1)
                cand_type = 2
            
            global_idx2 = chunk_start + idx_in_chunk
            
            if cand_type == 1:
                global_idx1 = int(pos_clipped[idx_in_chunk])
            else:
                global_idx1 = int(pos_minus[idx_in_chunk])
            
            best_idx1 = global_idx1
            best_idx2 = global_idx2
            
            if best_abs == 1:
                break
    
    if best_idx1 is None:
        # Fallback: simple nonzero sum
        return [0]*28 + [1, -1]
    
    # Unpack coefficients
    def unpack_coeffs(packed, n):
        res = [0] * n
        for i in range(n):
            val = (packed >> (2 * i)) & 3
            if val == 1:
                res[i] = 1
            elif val == 2:
                res[i] = -1
            else:
                res[i] = 0
        return res
    
    packed1 = int(coeffs1_sorted[best_idx1])
    packed2 = int(coeffs2[best_idx2])
    
    c1 = unpack_coeffs(packed1, 15)
    c2 = unpack_coeffs(packed2, 15)
    
    result = [0] * 30
    for i, idx in enumerate(half1):
        result[idx] = c1[i]
    for i, idx in enumerate(half2):
        result[idx] = c2[i]
    
    return result
