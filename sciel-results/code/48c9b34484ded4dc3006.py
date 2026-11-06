<<<<<<< SEARCH
def _objective_fast(a, b, A_eig, B_eig, H_eig, A_inv, B_inv, H_inv, dim):
    """Fast objective using precomputed eigendecompositions."""
    t_vals = [0.25, 0.5, 1.0]
    errors = []
    
    for t in t_vals:
        # Exact propagator
        exact = _expm_from_eig(H_eig[0], H_eig[1], H_inv, -1j * t)
        
        # Product formula: U = prod_k exp(-i b_k t B) exp(-i a_k t A)
        # Applied in order k = 0, 1, ..., 11
        U = np.eye(dim, dtype=complex)
        
        for k in range(len(a)):
            exp_a = _expm_from_eig(A_eig[0], A_eig[1], A_inv, -1j * a[k] * t)
            exp_b = _expm_from_eig(B_eig[0], B_eig[1], B_inv, -1j * b[k] * t)
            U = exp_b @ exp_a @ U
        
        diff = U - exact
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0] if len(s) > 0 else 0.0)
    
    return np.mean(errors)
=======
def _objective_fast(a, b, A_eig, B_eig, H_eig, A_inv, B_inv, H_inv, dim):
    """Fast objective using precomputed Hermitian eigendecompositions."""
    t_vals = [0.25, 0.5, 1.0]
    errors = []
    
    for t in t_vals:
        # Exact propagator
        exact = _expm_from_eigh(H_eig[0], H_eig[1], H_inv, -1j * t)
        
        # Product formula: U = prod_k exp(-i b_k t B) exp(-i a_k t A)
        # Applied in order k = 0, 1, ..., 11
        U = np.eye(dim, dtype=complex)
        
        for k in range(len(a)):
            exp_a = _expm_from_eigh(A_eig[0], A_eig[1], A_inv, -1j * a[k] * t)
            exp_b = _expm_from_eigh(B_eig[0], B_eig[1], B_inv, -1j * b[k] * t)
            U = exp_b @ exp_a @ U
        
        diff = U - exact
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0] if len(s) > 0 else 0.0)
    
    return np.mean(errors)
>>>>>>> REPLACE

