import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n_terms = 15
    # Generate sums and coefficients for first half (indices 0 to 14, terms 1 to 15)
    sums_A, coeffs_A = gen_sums_coeffs(n_terms, 0)
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]

    # Generate sums and coefficients for second half (indices 15 to 29, terms 16 to 30)
    sums_B, coeffs_B = gen_sums_coeffs(n_terms, 15)
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]

    # Sort sums_B and reorder coeffs_B
    idx_sort = np.argsort(sums_B)
    sums_B_sorted = sums_B[idx_sort]
    coeffs_B_sorted = coeffs_B[idx_sort]
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]

    # Vectorized search
    targets = -sums_A
    # searchsorted returns index where target would be inserted to maintain order
    # We want to check idx and idx-1
    idx_B = np.searchsorted(sums_B_sorted, targets)
    
    # Clip indices to valid range [0, len-1]
    len_B = len(sums_B_sorted)
    idx_B_clipped_0 = np.clip(idx_B, 0, len_B - 1)
    idx_B_clipped_1 = np.clip(idx_B - 1, 0, len_B - 1)
    
    # Calculate candidate sums
    # Candidate 0: using idx_B_clipped_0
    cand_sums_0 = sums_A + sums_B_sorted[idx_B_clipped_0]
    # Candidate 1: using idx_B_clipped_1
    cand_sums_1 = sums_A + sums_B_sorted[idx_B_clipped_1]
    
    # Combine into a 2D array for easier min finding
    # Shape: (len_A, 2)
    cand_sums = np.column_stack((cand_sums_0, cand_sums_1))
    
    # Calculate absolute values
    abs_cands = np.abs(cand_sums)
    
    # Mask out values close to zero (to enforce non-zero constraint)
    # Lower threshold to 1e-15 to capture smaller non-zero sums
    zero_mask = abs_cands < 1e-15
    abs_cands[zero_mask] = np.inf
    
    # Find the top K candidates to verify with exact arithmetic
    # K=50 is safe
    K = 50
    flat_indices = np.argpartition(abs_cands.ravel(), K)[:K]
    
    best_coeff = None
    best_val = float('inf')
    
    # Pre-compute exact denominators for terms 1..30
    # We need to check the exact sum for each candidate
    # Sum = sum(c_i / i)
    # We can represent this as a Fraction
    
    for idx in flat_indices:
        row = idx // 2
        col = idx % 2
        
        # Get coefficients
        best_a = coeffs_A[row]
        
        if col == 0:
            b_idx = idx_B_clipped_0[row]
        else:
            b_idx = idx_B_clipped_1[row]
            
        best_b = coeffs_B_sorted[b_idx]
        
        # Construct full coefficient vector
        full_coeffs = np.concatenate([best_a, best_b])
        
        # Calculate exact sum
        # Optimization: Only compute if float sum is small enough
        # The float sum is abs_cands[row, col]
        # If it's already larger than current best_val, skip?
        # But best_val is in float space. We need exact comparison.
        # Let's just compute for the top K. K is small.
        
        s = Fraction(0, 1)
        for i in range(30):
            c = int(full_coeffs[i])
            if c != 0:
                s += Fraction(c, i + 1)
        
        if s == 0:
            continue
            
        abs_s = abs(s)
        
        if abs_s < best_val:
            best_val = abs_s
            best_coeff = full_coeffs
            
        if time.time() > deadline:
            break

    if best_coeff is None:
        # Fallback
        return [0]*28 + [1, -1]
        
    return [int(x) for x in best_coeff]

def gen_sums_coeffs(n_terms, start_idx):
    sums = np.array([0.0])
    coeffs = np.zeros((1, n_terms), dtype=np.int8)
    
    for i in range(n_terms):
        term = 1.0 / (start_idx + i + 1)
        
        # Create new arrays of size 3 * current_size
        new_size = len(sums) * 3
        new_sums = np.empty(new_size)
        new_coeffs = np.empty((new_size, n_terms), dtype=np.int8)
        
        # 1. Coeff 0
        new_sums[:len(sums)] = sums
        new_coeffs[:len(sums)] = coeffs
        
        # 2. Coeff 1
        new_sums[len(sums):2*len(sums)] = sums + term
        new_coeffs[len(sums):2*len(sums)] = coeffs
        new_coeffs[len(sums):2*len(sums), i] = 1
        
        # 3. Coeff -1
        new_sums[2*len(sums):] = sums - term
        new_coeffs[2*len(sums):] = coeffs
        new_coeffs[2*len(sums):, i] = -1
        
        sums = new_sums
        coeffs = new_coeffs
        
    return sums, coeffs
