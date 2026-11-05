def _structured_init(R, terms, seed=42):
    """Create a structured initialization based on constraint grouping."""
    n, m, p = DIMENSIONS
    u_rows = n * m
    v_rows = m * p
    w_rows = p * n
    
    np.random.seed(seed)
    
    # Group constraints by shared structure
    # Two constraints (i1,j1,k1) and (i2,j2,k2) can potentially share a column
    # if they have overlapping U, V, or W indices
    
    # Build groups greedily
    groups = []
    used = [False] * len(terms)
    
    for idx in range(len(terms)):
        if used[idx]:
            continue
        group = [idx]
        used[idx] = True
        u_idx, v_idx, w_idx = terms[idx]
        
        for jdx in range(idx + 1, len(terms)):
            if used[jdx]:
                continue
            ju_idx, jv_idx, jw_idx = terms[jdx]
            # Check if they share any index
            if u_idx == ju_idx or v_idx == jv_idx or w_idx == jw_idx:
                group.append(jdx)
                used[jdx] = True
    
        groups.append(group)
    
    # If we have more groups than R, merge some
    while len(groups) > R:
        # Find the two smallest groups and merge them
        groups.sort(key=len)
        g1, g2 = groups[0], groups[1]
        merged = g1 + g2
        groups = [merged] + groups[2:]
    
    # Pad with empty groups if needed
    while len(groups) < R:
        groups.append([])
    
    # Create initialization
    U = np.zeros((u_rows, R))
    V = np.zeros((v_rows, R))
    W = np.zeros((w_rows, R))
    
    for r, group in enumerate(groups):
        for idx in group:
            u_idx, v_idx, w_idx = terms[idx]
            U[u_idx, r] = np.random.randn() * 0.3
            V[v_idx, r] = np.random.randn() * 0.3
            W[w_idx, r] = np.random.randn() * 0.3
    
    return U, V, W
