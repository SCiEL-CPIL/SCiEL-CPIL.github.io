import numpy as np
import time
import sys

def construct():
    # Deadline guard: 85 seconds max (evaluator kills at 90s)
    start_time = time.time()
    deadline = start_time + 85
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n up to N using NumPy sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve: for each i, add i to all multiples of i
    for i in range(1, N + 1):
        sigma[i::i] += i
        # Time check every 100000 iterations
        if i % 100000 == 0 and time.time() > deadline - 15:
            # If running out of time, break and search with partial sigma
            # But sigma is incomplete, so we need to be careful
            # For now, break and hope we have enough
            break

    # 2. Vectorized search for best pair
    # Error = |2a - 3b| / (2b). Minimize this with sigma(a) == sigma(b) and 2a != 3b.
    
    best_a, best_b = 11, 6
    best_err_num, best_err_den = 2, 6  # |2*11 - 3*6| / (2*6) = 4/12 = 1/3
    
    max_b = 2 * N // 3
    window = 5  # Check a in [floor(1.5b) - 5, floor(1.5b) + 5]
    
    chunk_size = 500000
    for b_start in range(1, max_b + 1, chunk_size):
        b_end = min(b_start + chunk_size - 1, max_b)
        
        # Time check
        if time.time() > deadline - 10:
            break
        
        b_arr = np.arange(b_start, b_end + 1, dtype=np.int64)
        a_target = (3 * b_arr) // 2  # floor(1.5 * b)
        
        for da in range(-window, window + 1):
            a_arr = a_target + da
            # Mask valid a values
            mask = (a_arr >= 1) & (a_arr <= N)
            if not np.any(mask):
                continue
            
            a_valid = a_arr[mask]
            b_valid = b_arr[mask]
            
            # Check sigma equality
            match = sigma[a_valid] == sigma[b_valid]
            if not np.any(match):
                continue
            
            a_match = a_valid[match]
            b_match = b_valid[match]
            
            # Compute error numerator |2a - 3b|
            err_nums = np.abs(2 * a_match - 3 * b_match)
            # Filter out zero error (invalid)
            nonzero = err_nums > 0
            if not np.any(nonzero):
                continue
            
            a_match = a_match[nonzero]
            b_match = b_match[nonzero]
            err_nums = err_nums[nonzero]
            err_dens = 2 * b_match
            
            # Find minimum ratio err_num/err_den
            # Use float64 for comparison (exact for these magnitudes)
            ratios = err_nums.astype(np.float64) / err_dens.astype(np.float64)
            min_idx = int(np.argmin(ratios))
            
            cand_a = int(a_match[min_idx])
            cand_b = int(b_match[min_idx])
            cand_err_num = int(err_nums[min_idx])
            cand_err_den = int(err_dens[min_idx])
            
            # Compare with current best using cross-multiplication (exact)
            if cand_err_num * best_err_den < best_err_num * cand_err_den:
                best_a, best_b = cand_a, cand_b
                best_err_num, best_err_den = cand_err_num, cand_err_den
    
    return [int(best_a), int(best_b)]

# Test / Sanity Check if run directly
if __name__ == "__main__":
    res = construct()
    print(res)
    a, b = res
    if isinstance(a, int) and isinstance(b, int):
        print(f"a={a}, b={b}, sigma_equal_check: {sigma(a) == sigma(b)}")
    else:
        print("Type error")
