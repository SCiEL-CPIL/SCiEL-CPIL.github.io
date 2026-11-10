import numpy as np
import time

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
    # Threshold 1e-12 is safe as min non-zero is ~4e-13
    zero_mask = abs_cands < 1e-12
    abs_cands[zero_mask] = np.inf
    
    # Find global minimum
    min_val = np.min(abs_cands)
    
    if np.isinf(min_val):
        # Fallback if no non-zero sum found (should not happen)
        return [0]*28 + [1, -1]
        
    min_idx = np.argmin(abs_cands)
    row = min_idx // 2
    col = min_idx % 2
    
    best_a = coeffs_A[row]
    
    if col == 0:
        b_idx = idx_B_clipped_0[row]
    else:
        b_idx = idx_B_clipped_1[row]
        
    best_b = coeffs_B_sorted[b_idx]
    
    result = np.concatenate([best_a, best_b])
    return [int(x) for x in result]

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
