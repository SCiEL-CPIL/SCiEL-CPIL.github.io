<<<<<<< SEARCH
def _random_search_for_rank(T, nm, mp, pn, rank, max_iters=500, seed=42):
    """Try to find a rank-r decomposition using random init + ALS refinement."""
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(nm, rank) * 0.1
    V = rng.randn(mp, rank) * 0.1
    W = rng.randn(pn, rank) * 0.1
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute current tensor
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-9:
            break
        
        # ALS: update U by fixing V, W
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed V,W: T[a].ravel() = U[a,:] @ B_flat where B_flat[b*pn+c, r] = V[b,r]*W[c,r]
        B_flat = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, rank)
        B_pinv = np.linalg.pinv(B_flat)
        
        # Vectorized update of U
        T_flat = T.reshape(nm, mp * pn)
        U = T_flat @ B_pinv
        
        # Update V
        A_flat = np.einsum('ar,cr->acr', U, W).reshape(nm * pn, rank)
        A_pinv = np.linalg.pinv(A_flat)
        
        T_reshape = T.reshape(mp, nm * pn)
        V = T_reshape @ A_pinv
        
        # Update W
        C_flat = np.einsum('ar,br->abr', U, V).reshape(nm * mp, rank)
        C_pinv = np.linalg.pinv(C_flat)
        
        T_reshape2 = T.reshape(pn, nm * mp)
        W = T_reshape2 @ C_pinv
    
    return best_U, best_V, best_W, best_err
=======
def _random_search_for_rank(T, nm, mp, pn, rank, max_iters=500, seed=42, deadline=140.0, start_time=0.0):
    """Try to find a rank-r decomposition using random init + ALS refinement."""
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(nm, rank) * 0.1
    V = rng.randn(mp, rank) * 0.1
    W = rng.randn(pn, rank) * 0.1
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > deadline:
            break
        
        # Compute current tensor
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-9:
            break
        
        # ALS: update U by fixing V, W
        B_flat = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, rank)
        B_pinv = np.linalg.pinv(B_flat)
        
        T_flat = T.reshape(nm, mp * pn)
        U = T_flat @ B_pinv
        
        # Update V
        A_flat = np.einsum('ar,cr->acr', U, W).reshape(nm * pn, rank)
        A_pinv = np.linalg.pinv(A_flat)
        
        T_for_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
        V = T_for_V @ A_pinv
        
        # Update W
        C_flat = np.einsum('ar,br->abr', U, V).reshape(nm * mp, rank)
        C_pinv = np.linalg.pinv(C_flat)
        
        T_for_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
        W = T_for_W @ C_pinv
    
    return best_U, best_V, best_W, best_err
>>>>>>> REPLACE

