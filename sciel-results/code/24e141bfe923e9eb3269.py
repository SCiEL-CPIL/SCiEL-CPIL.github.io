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
    
    # Generate all 3^15 signed sums for first half
    def gen_all_sums(A_part):
        n = len(A_part)
        sums = np.array([0], dtype=np.int64)
        for i in range(n):
            a_i = A_part[i]
            sums = np.concatenate([sums + a_i, sums, sums - a_i])
        return sums
    
    sums1 = gen_all_sums(A1)
    sums1.sort()
    
    # Generate all 3^15 signed sums for second half
    sums2 = gen_all_sums(A2)
    
    # For each s2, find s1 such that s1 + s2 is closest to 0 but not 0
    best_abs = None
    best_s1 = None
    best_s2 = None
    
    n1 = len(sums1)
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
        pos = np.searchsorted(sums1, targets)
        
        # Candidate 1: sums1[pos] (clipped)
        pos_clipped = np.clip(pos, 0, n1 - 1)
        s1_cand1 = sums1[pos_clipped]
        total1 = s1_cand1 + s2_chunk
        abs_total1 = np.abs(total1)
        
        # Candidate 2: sums1[pos-1] (clipped)
        pos_minus = np.clip(pos - 1, 0, n1 - 1)
        s1_cand2 = sums1[pos_minus]
        total2 = s1_cand2 + s2_chunk
        abs_total2 = np.abs(total2)
        
        # Combine and find minimum nonzero
        all_abs = np.concatenate([abs_total1, abs_total2])
        all_s1 = np.concatenate([s1_cand1, s1_cand2])
        all_s2 = np.concatenate([s2_chunk, s2_chunk])
        all_total = np.concatenate([total1, total2])
        
        # Mask out zeros
        nonzero_mask = all_total != 0
        if not np.any(nonzero_mask):
            continue
        
        abs_nonzero = np.where(nonzero_mask, all_abs, np.iinfo(np.int64).max)
        min_idx = np.argmin(abs_nonzero)
        min_abs = int(abs_nonzero[min_idx])
        
        if best_abs is None or min_abs < best_abs:
            best_abs = min_abs
            best_s1 = int(all_s1[min_idx])
            best_s2 = int(all_s2[min_idx])
            
            if best_abs == 1:
                break
    
    if best_s1 is None:
        # Fallback: simple nonzero sum
        return [0]*28 + [1, -1]
    
    # Reconstruct coefficients
    def find_coeffs(target, A_part):
        n = len(A_part)
        coeffs = [0] * n
        
        def dfs(idx, current_sum):
            if idx == n:
                return current_sum == target
            
            a_i = A_part[idx]
            rem = target - current_sum
            
            # Try values in order of proximity to remaining target
            if rem > 0:
                order = [1, 0, -1]
            elif rem < 0:
                order = [-1, 0, 1]
            else:
                order = [0, 1, -1]
            
            for val in order:
                coeffs[idx] = val
                if dfs(idx + 1, current_sum + val * a_i):
                    return True
            
            return False
        
        if dfs(0, 0):
            return coeffs
        return None
    
    c1 = find_coeffs(best_s1, A1)
    c2 = find_coeffs(best_s2, A2)
    
    if c1 is None or c2 is None:
        return [0]*28 + [1, -1]
    
    result = [0] * 30
    for i, idx in enumerate(half1):
        result[idx] = c1[i]
    for i, idx in enumerate(half2):
        result[idx] = c2[i]
    
    return result
