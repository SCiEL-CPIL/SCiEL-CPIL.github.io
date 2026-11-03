import numpy as np

def _score_fast(U):
    if not U or max(U) == 0:
        return 1.0
    U_arr = np.array(U, dtype=np.int64)
    # Ensure 0 is present and unique
    # U+U
    sums = np.add.outer(U_arr, U_arr)
    unique_sums = np.unique(sums)
    # U-U
    diffs = np.subtract.outer(U_arr, U_arr)
    unique_diffs = np.unique(diffs)
    
    len_sum = len(unique_sums)
    len_diff = len(unique_diffs)
    
    if len_sum == 0: return 1.0
    
    ratio = len_diff / len_sum
    if ratio <= 1:
        return 1.0
        
    max_u = max(U_arr)
    log_ratio = np.log(ratio)
    log_denom = np.log(2 * max_u + 1)
    
    return 1.0 + log_ratio / log_denom
