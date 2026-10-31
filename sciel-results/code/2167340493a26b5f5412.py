def _get_max_overlap(x, y, r, i):
    max_ov = 0.0
    max_idx = -1
    
    ov = r[i] - x[i]
    if ov > max_ov:
        max_ov = ov
        max_idx = 0
    ov = r[i] - (1.0 - x[i])
    if ov > max_ov:
        max_ov = ov
        max_idx = 1
    ov = r[i] - y[i]
    if ov > max_ov:
        max_ov = ov
        max_idx = 2
    ov = r[i] - (1.0 - y[i])
    if ov > max_ov:
        max_ov = ov
        max_idx = 3

    for j in range(N):
        if i == j: continue
        d = np.hypot(x[i] - x[j], y[i] - y[j])
        ov = r[i] + r[j] - d
        if ov > max_ov:
            max_ov = ov
            max_idx = j
            
    return max_ov, max_idx
