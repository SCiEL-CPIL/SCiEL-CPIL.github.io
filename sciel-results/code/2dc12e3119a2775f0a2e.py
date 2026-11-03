def _try_random_projection_reduce(U, V, W, n, m, p, target_rank, deadline, tol=1e-9):
    """
    Try to reduce rank to target_rank by projecting onto a random subspace.
    """
    nm, mp, pn = n * m, m * p, p * n
    r_curr = U.shape[1]
    
    if target_rank >= r_curr:
        return None
        
    T_target = _build_target_tensor(n, m, p)
    
    # Try a few random projections
    for _ in range(5):
        if time.time() > deadline - 10:
            break
            
        # Generate random projection matrices
        # We want U_new = U @ P_U, where P_U is (r_curr, target_rank)
        # To maintain scale, we can use Gaussian random matrices.
        P_U = np.random.randn(r_curr, target_rank)
        P_V = np.random.randn(r_curr, target_rank)
        P_W = np.random.randn(r_curr, target_rank)
        
        U_new = U @ P_U
        V_new = V @ P_V
        W_new = W @ P_W
        
        # Refine with ALS
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            U_new = _als_update_U(U_new, V_new, W_new, T_target, n, m, p, lam=1e-6)
            V_new = _als_update_V(U_new, V_new, W_new, T_target, n, m, p, lam=1e-6)
            W_new = _als_update_W(U_new, V_new, W_new, T_target, n, m, p, lam=1e-6)
            
        if _verify_decomp(n, m, p, U_new, V_new, W_new, tol):
            return U_new, V_new, W_new
            
    return None
