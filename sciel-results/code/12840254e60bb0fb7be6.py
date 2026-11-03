<<<<<<< SEARCH
def _try_better_decomposition(deadline):
    """
    Try to find a decomposition with rank < 84 using ALS and rank-1 peeling.
    Strategy:
    1. Start from the block decomposition (rank 84).
    2. Use ALS to refine the decomposition at various target ranks.
    3. Use rank-1 peeling: identify the dominant rank-1 component via SVD,
       subtract it, and recurse on the residual.
    """
    import numpy as np
    n, m, p = 3, 5, 6
    T = _build_target_tensor(n, m, p)
    T_flat = T.reshape(n*m*m*p, p*n)  # unfold over (a,b) and c
    
    best_rank = 90  # schoolbook
    best_result = None
    
    # Try various target ranks using ALS
    for target_rank in range(83, 60, -1):
        if time.time() > deadline - 20:
            break
        
        # Initialize from block decomposition, truncated
        U_init, V_init, W_init = _block_decomposition_356()
        
        # Truncate to target_rank by keeping first target_rank columns
        # (the block decomposition is ordered, so early blocks are "important")
        U_r = U_init[:, :target_rank].copy()
        V_r = V_init[:, :target_rank].copy()
        W_r = W_init[:, :target_rank].copy()
        
        # ALS iterations
        converged = False
        for it in range(50):
            if time.time() > deadline - 15:
                break
            
            # Solve for W: for each c, W[c, :] should satisfy
            # T[:, :, c] = sum_k U[:, k] * V[:, k] * W[c, k]
            # i.e., T_c = (U * V_k) * W_c  where U*V is element-wise per k
            # Reshape: T[:, :, c] is (n*m, m*p)
            # We want: T[:, :, c] = sum_k W[c,k] * (U[:, k] * V[:, k])
            # So for fixed c: T[:,:,c].reshape(-1) = sum_k W[c,k] * (U[:,k]*V[:,k]).reshape(-1)
            # This is a linear system: M_c * W[:, c] = T[:,:,c].reshape(-1)
            # where M_c has columns (U[:,k]*V[:,k]).reshape(-1)
            
            # Build the design matrix once
            # D[a, b, k] = U[a, k] * V[b, k]
            D = U_r[:, :, None] * V_r[None, :, :]  # (n*m, m*p, rank)
            D_flat = D.reshape(n*m*m*p, target_rank)  # (n*m*m*p, rank)
            
            # T_flat is (n*m*m*p, p*n)
            # Solve: D_flat @ W = T_flat  (least squares)
            W_r, residuals, rank_d, sv = np.linalg.lstsq(D_flat, T_flat, rcond=None)
            
            # Solve for U: for each a, U[a, :] should satisfy
            # T[a, :, :] = sum_k U[a,k] * V[:, k] * W[:, k]
            # T[a, b, c] = sum_k U[a,k] * (V[b,k] * W[c,k])
            # For fixed a: T[a, :, :].reshape(-1) = sum_k U[a,k] * (V[:,k]*W[:,k]).reshape(-1)
            D_U = (V_r[:, :, None] * W_r[None, :, :]).reshape(m*p*p*n, target_rank)
            T_U = T.reshape(n, m*p*p*n)
            U_r, _, _, _ = np.linalg.lstsq(D_U, T_U, rcond=None)
            
            # Solve for V: for each b, V[b, :] should satisfy
            # T[:, b, :] = sum_k V[b,k] * U[:, k] * W[:, k]
            # T[a, b, c] = sum_k V[b,k] * (U[a,k] * W[c,k])
            D_V = (U_r[:, :, None] * W_r[None, :, :]).reshape(n*m*p*n, target_rank)
            T_V = T.reshape(m, n*m*p*n)
            V_r, _, _, _ = np.linalg.lstsq(D_V, T_V, rcond=None)
        
        # Verify
        if _verify_decomp(n, m, p, U_r, V_r, W_r):
            best_rank = target_rank
            best_result = (U_r, V_r, W_r)
            break
        elif time.time() > deadline - 10:
            break
    
    if best_result is not None:
        return best_result
    return None
=======
def _try_better_decomposition(deadline):
    """
    Try to find a decomposition with rank < 84 using ALS.
    Uses alternating least squares to solve for U, V, W at target ranks.
    """
    n, m, p = 3, 5, 6
    T = _build_target_tensor(n, m, p)
    # T shape: (n*m, m*p, p*n) = (15, 30, 18)
    
    best_result = None
    best_rank = 90
    
    # Precompute reshapes for efficient ALS
    T_W_flat = np.transpose(T, (2, 0, 1)).reshape(p*n, n*m*m*p)  # (18, 450)
    T_U_flat = T.reshape(n*m, m*p*p*n)  # (15, 540)
    T_V_flat = np.transpose(T, (1, 0, 2)).reshape(m*p, n*m*p*n)  # (30, 270)
    
    # Get initial decomposition
    U_init, V_init, W_init = _block_decomposition_356()
    init_rank = U_init.shape[1]  # 84
    
    # Try target ranks from just below current best downward
    for target_rank in range(init_rank - 1, max(init_rank - 20, 50), -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try a few initializations
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            if attempt == 0:
                # Initialize from truncated block decomposition
                U_r = U_init[:, :target_rank].astype(np.float64).copy()
                V_r = V_init[:, :target_rank].astype(np.float64).copy()
                W_r = W_init[:, :target_rank].astype(np.float64).copy()
            else:
                # Random initialization with small perturbation
                U_r = U_init[:, :target_rank].astype(np.float64).copy()
                V_r = V_init[:, :target_rank].astype(np.float64).copy()
                W_r = W_init[:, :target_rank].astype(np.float64).copy()
                noise = 0.01 * np.random.randn(target_rank)
                U_r += 0.01 * np.random.randn(n*m, target_rank)
                V_r += 0.01 * np.random.randn(m*p, target_rank)
                W_r += 0.01 * np.random.randn(p*n, target_rank)
            
            # ALS iterations
            for it in range(40):
                if time.time() > deadline - 8:
                    break
                
                # Solve for W: T_W_flat = W_r @ D_W
                D_W = (U_r[:, :, None] * V_r[None, :, :]).reshape(target_rank, n*m*m*p)
                W_new, _, _, _ = np.linalg.lstsq(D_W.T, T_W_flat.T, rcond=None)
                W_r = W_new.T
                
                # Solve for U: T_U_flat = U_r @ D_U
                D_U = (V_r[:, :, None] * W_r[None, :, :]).reshape(target_rank, m*p*p*n)
                U_new, _, _, _ = np.linalg.lstsq(D_U.T, T_U_flat.T, rcond=None)
                U_r = U_new.T
                
                # Solve for V: T_V_flat = V_r @ D_V
                D_V = (U_r[:, :, None] * W_r[None, :, :]).reshape(target_rank, n*m*p*n)
                V_new, _, _, _ = np.linalg.lstsq(D_V.T, T_V_flat.T, rcond=None)
                V_r = V_new.T
            
            # Verify
            if _verify_decomp(n, m, p, U_r, V_r, W_r):
                best_rank = target_rank
                best_result = (U_r, V_r, W_r)
                found = True
                break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    return None
>>>>>>> REPLACE
