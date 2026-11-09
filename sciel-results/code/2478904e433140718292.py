import time
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 30
    # LCM of 1..30
    L = 1
    for i in range(1, N + 1):
        L = L * i // gcd(L, i)
    
    # A[i] = L // i, so sum c[i-1]/i = (sum c[i-1]*A[i]) / L
    A = np.array([L // i for i in range(1, N + 1)], dtype=np.int64)
    
    # Split into two halves of 15
    half1_idx = np.arange(0, 15)
    half2_idx = np.arange(15, 30)
    
    A1 = A[half1_idx]
    A2 = A[half2_idx]
    
    # Generate all 3^15 signed sums for each half using NumPy
    def gen_all_sums(A_part):
        n = len(A_part)
        # Start with [0]
        sums = np.array([0], dtype=np.int64)
        for i in range(n):
            a_i = A_part[i]
            # For each existing sum, create 3 new sums: +a_i, 0, -a_i
            sums = np.concatenate([sums + a_i, sums, sums - a_i])
        return sums
    
    sums1 = gen_all_sums(A1)
    sums2 = gen_all_sums(A2)
    
    # Sort sums1 for binary search
    sums1.sort()
    n1 = len(sums1)
    
    # For each s2, find s1 such that s1 + s2 is closest to 0 but not 0
    # The minimum |s1+s2| for nonzero is what we want
    # Since we want the smallest nonzero |total|, and total = s1+s2,
    # we want s1 close to -s2 but s1 != -s2 (to avoid total=0)
    
    best_abs = None
    best_s1 = None
    best_s2 = None
    
    # Use numpy vectorized operations for speed
    # For each s2, target = -s2
    # Find position in sorted sums1
    targets = -sums2
    positions = np.searchsorted(sums1, targets)
    
    # Clip positions to valid range
    positions_clipped = np.clip(positions, 0, n1 - 1)
    
    # Candidate 1: sums1[positions]
    # Candidate 2: sums1[positions-1] (if positions > 0)
    
    # For each s2, check the two nearest s1 values
    # We need to be careful about memory, so process in chunks
    
    chunk_size = 100000
    n2 = len(sums2)
    
    for chunk_start in range(0, n2, chunk_size):
        if time.time() > deadline - 10:
            break
        
        chunk_end = min(chunk_start + chunk_size, n2)
        s2_chunk = sums2[chunk_start:chunk_end]
        targets_chunk = -s2_chunk
        pos_chunk = np.searchsorted(sums1, targets_chunk)
        
        # Candidate indices
        pos_clipped = np.clip(pos_chunk, 0, n1 - 1)
        
        # Get candidate s1 values
        s1_candidates = np.empty((len(s2_chunk), 2), dtype=np.int64)
        s1_candidates[:, 0] = sums1[pos_clipped]
        
        # For the second candidate (position - 1), handle boundaries
        pos_minus = np.clip(pos_chunk - 1, 0, n1 - 1)
        s1_candidates[:, 1] = sums1[pos_minus]
        
        # Compute totals
        totals = np.empty((len(s2_chunk), 2), dtype=np.int64)
        totals[:, 0] = s1_candidates[:, 0] + s2_chunk
        totals[:, 1] = s1_candidates[:, 1] + s2_chunk
        
        # Filter out zeros
        abs_totals = np.abs(totals)
        
        # Find minimum nonzero
        mask = abs_totals > 0
        if not np.any(mask):
            continue
        
        # Replace zeros with infinity
        abs_totals_masked = np.where(mask, abs_totals, np.iinfo(np.int64).max)
        
        min_idx = np.argmin(abs_totals_masked)
        row = min_idx // 2
        col = min_idx % 2
        
        min_abs = abs_totals_masked[row, col]
        
        if best_abs is None or min_abs < best_abs:
            best_abs = int(min_abs)
            best_s1 = int(s1_candidates[row, col])
            best_s2 = int(s2_chunk[row])
            
            if best_abs == 1:
                # Can't do better than 1
                break
    
    if best_s1 is None:
        # Fallback: return a simple nonzero sum
        return [0]*28 + [1, -1]
    
    # Now reconstruct the coefficients
    # We need to find c1 such that sum(c1[i]*A1[i]) = best_s1
    # and c2 such that sum(c2[i]*A2[i]) = best_s2
    
    def find_coeffs(target, A_part):
        n = len(A_part)
        coeffs = [0] * n
        
        def dfs(idx, current_sum):
            if idx == n:
                return current_sum == target
            
            a_i = A_part[idx]
            
            # Try 1
            coeffs[idx] = 1
            if dfs(idx + 1, current_sum + a_i):
                return True
            
            # Try 0
            coeffs[idx] = 0
            if dfs(idx + 1, current_sum):
                return True
            
            # Try -1
            coeffs[idx] = -1
            if dfs(idx + 1, current_sum - a_i):
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
    for i in range(15):
        result[i] = c1[i]
    for i in range(15):
        result[15 + i] = c2[i]
    
    return result
