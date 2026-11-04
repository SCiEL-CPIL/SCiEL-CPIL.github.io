def score_set_numpy(U):
    U = np.array(sorted(set(U)), dtype=np.int64)
    if 0 not in U: return -1
    n = len(U)
    if n < 2: return -1
    
    # Sumset
    # U + U
    sums = U[:, None] + U[None, :]
    unique_sums = np.unique(sums)
    sum_count = len(unique_sums)
    
    # Diffset
    # U - U
    diffs = U[:, None] - U[None, :]
    unique_diffs = np.unique(diffs)
    diff_count = len(unique_diffs)
    
    if sum_count == 0: return -1
    ratio = diff_count / sum_count
    if ratio <= 1: return -1
    
    max_u = np.max(U)
    D = 2 * max_u + 1
    if D <= 1: return -1
    
    return 1 + np.log(ratio) / np.log(D)
