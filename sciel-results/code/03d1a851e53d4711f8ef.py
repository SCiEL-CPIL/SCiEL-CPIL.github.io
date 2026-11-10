def _compute_area(x):
    n = 64
    num_heights = 20000
    
    a = (np.arange(n) + 1) / n
    y = (np.arange(num_heights) + 0.5) / num_heights
    w = (1.0 - y) / n
    
    x_diff = x[None, :] - x[:, None]
    a_diff = a[None, :] - a[:, None]
    y_cross = x_diff / (-a_diff)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    y_cross_valid = y_cross[i_idx, j_idx]
    pairs = np.column_stack((i_idx, j_idx))
    
    mask = (y_cross_valid > 0) & (y_cross_valid < 1)
    y_cross_valid = y_cross_valid[mask]
    pairs = pairs[mask]
    
    sort_order = np.argsort(y_cross_valid)
    y_cross_sorted = y_cross_valid[sort_order]
    pairs_sorted = pairs[sort_order]
    
    order = np.argsort(x)
    inv_order = np.empty(n, dtype=int)
    inv_order[order] = np.arange(n)
    
    U = np.zeros(num_heights)
    
    cross_idx = 0
    num_crossings = len(y_cross_sorted)
    
    for m in range(num_heights):
        y_m = y[m]
        
        while cross_idx < num_crossings and y_cross_sorted[cross_idx] <= y_m:
            i, j = pairs_sorted[cross_idx]
            pos_i = inv_order[i]
            pos_j = inv_order[j]
            
            order[pos_i], order[pos_j] = order[pos_j], order[pos_i]
            inv_order[i], inv_order[j] = pos_j, pos_i
            cross_idx += 1
            
        L_m = x + a * y_m
        L_sorted = L_m[order]
        
        d = np.diff(L_sorted)
        w_m = w[m]
        gap_contrib = np.sum(np.minimum(d, w_m))
        U[m] = w_m + gap_contrib
        
    return np.mean(U)
