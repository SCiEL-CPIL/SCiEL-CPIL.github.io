def _check_full_residual_vec(U, V, W, non_zero_set, n, m, p):
    R = len(U[0])
    # Compute T
    T = np.zeros((n*m, m*p, p*n))
    for r in range(R):
        T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
    # Flatten T
    T_flat = T.flatten()
    
    # Check non-zero
    for u, v, w in non_zero_set:
        idx = (u * (m*p) + v) * (p*n) + w
        if abs(T_flat[idx] - 1.0) > 1e-9:
            return False
            
    # Check zero
    # Create a mask of non-zero indices
    mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for u, v, w in non_zero_set:
        mask[u, v, w] = True
        
    # Check if any zero entry is non-zero
    zero_entries = T_flat[~mask.flatten()]
    if np.max(np.abs(zero_entries)) > 1e-9:
        return False
        
    return True
