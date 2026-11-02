def _compute_error_fast(U, V, W, T):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    D = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        
        # D[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
        # Use einsum or manual
        # For each i: D[i] += u_c[i,:,None] * (v_c * w_c[:,i][None,:])
        for i in range(n):
            D[i] += u_c[i, :, None] * (v_c * w_c[:, i][None, :])
    
    return float(np.sum((D - T) ** 2))
