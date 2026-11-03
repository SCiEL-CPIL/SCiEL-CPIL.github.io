def _decomp_4x4x3_pruned():
    """
    Decomposition of M_{4,4,3} by pruning M_{4,4,4}.
    """
    n, m, p = 4, 4, 4
    U, V, W = _decomp_4x4x4()
    
    # W shape: (p*n, rank) -> (4*4, 49) = (16, 49)
    # Index: k*n + i
    # We want to keep columns k=0,1,2.
    # A term r is "useless" for 4x4x3 if W[k*n+i, r] == 0 for all i and k in {0,1,2}.
    
    keep = []
    for r in range(U.shape[1]):
        is_useful = False
        for k in range(3):
            for i in range(n):
                if W[k * n + i, r] != 0:
                    is_useful = True
                    break
            if is_useful:
                break
        if is_useful:
            keep.append(r)
            
    U_pruned = U[:, keep]
    V_pruned = V[:, keep]
    W_pruned = W[:, keep]
    
    # Now we need to adjust V and W to be for p=3.
    # V shape: (m*p, rank) -> (4*4, rank). We only need rows for k=0,1,2.
    # V index: j*p + k.
    # We need to create V_new of shape (m*3, len(keep)).
    V_new = np.zeros((m * 3, len(keep)))
    for j in range(m):
        for k in range(3):
            V_new[j * 3 + k, :] = V[j * 4 + k, keep]
            
    # W shape: (p*n, rank). We only need rows for k=0,1,2.
    # W index: k*n + i.
    W_new = np.zeros((3 * n, len(keep)))
    for k in range(3):
        for i in range(n):
            W_new[k * n + i, :] = W[k * n + i, keep]
            
    return U_pruned, V_new, W_new
