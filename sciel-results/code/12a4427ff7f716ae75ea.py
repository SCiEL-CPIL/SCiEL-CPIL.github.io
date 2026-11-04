def _try_reduce_rank(U, V, W, n, m, p, rank, deadline):
    """
    Try to reduce the rank by dropping columns and adjusting the rest.
    Returns (best_U, best_V, best_W, best_rank) as lists.
    """
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    best_rank = rank
    
    # 1. Check for trivial zero columns (or near-zero)
    # If a column is all zeros, we can just drop it.
    # However, Strassen factors usually don't have all-zero columns.
    # We check if any column has norm < 1e-10.
    for r in range(rank):
        if np.linalg.norm(Ua[:, r]) < 1e-10 or np.linalg.norm(Va[:, r]) < 1e-10 or np.linalg.norm(Wa[:, r]) < 1e-10:
             # Drop this column
             keep = [i for i in range(rank) if i != r]
             Ua = Ua[:, keep]
             Va = Va[:, keep]
             Wa = Wa[:, keep]
             best_rank -= 1
             best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
             # Re-check? Usually only one. Break for safety in loop context.
             break 

    if best_rank == rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank

    # 2. Try to reduce by 1 more via optimization (if time permits)
    # We now have best_rank. Try to go to best_rank - 1.
    current_rank = best_rank
    if time.time() > deadline - 15:
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank

    target_rank = current_rank - 1
    
    # We need to find u, v, w such that sum_{r=0}^{target_rank-1} U_r V_r W_r + u v w = T
    # where T is the target tensor (all ones for matmul).
    # This is equivalent to finding u, v, w such that u v w = T - sum_{r=0}^{target_rank-1} U_r V_r W_r
    # Let R = T - sum_{r=0}^{target_rank-1} U_r V_r W_r
    # We want rank(R) <= 1.
    
    # We will try to optimize the LAST column (index target_rank-1) of the current best_U/V/W 
    # assuming the first target_rank-1 columns are fixed.
    # But which columns to fix? The ones from the original rank-77 set, excluding the one we want to "absorb" or drop?
    # Actually, the standard approach: Take the first target_rank-1 columns of the current best_U/V/W.
    # Compute residual R.
    # Optimize the (target_rank)-th column to match R.
    
    U_base = best_U[:, :target_rank]
    V_base = best_V[:, :target_rank]
    W_base = best_W[:, :target_rank]
    
    # Compute Residual R = T - sum_{r in base} U_r V_r W_r
    # T[i,j,k] = 1.
    # R[i,j,k] = 1 - sum_{r} U[i,j,r] V[j,k,r] W[k,i,r]
    # Dimensions: n*m, m*p, p*n
    # U_base shape: (n*m, target_rank)
    # V_base shape: (m*p, target_rank)
    # W_base shape: (p*n, target_rank)
    
    # Reshape for einsum
    # Ur[i, j, r] -> U_base[i*m+j, r]
    # We can compute the sum directly.
    
    # To save memory and time, we compute R as a 3D array.
    # R has size n*m * m*p * p*n = 16 * 24 * 24 = 9216 elements.
    # This is small enough to keep in memory.
    
    R = np.ones((n*m, m*p, p*n), dtype=np.float64)
    
    # Subtract contributions from base columns
    for r in range(target_rank):
        u = U_base[:, r]
        v = V_base[:, r]
        w = W_base[:, r]
        # term[i, j, k] = u[i] * v[j] * w[k]
        # We need to index correctly.
        # U is indexed by i*m+j.
        # V is indexed by j*p+k.
        # W is indexed by k*n+i.
        
        # Let's construct the term tensor.
        # term[i, j, k] = U_base[i*m+j, r] * V_base[j*p+k, r] * W_base[k*n+i, r]
        
        # We can do this with loops over i, j, k? No, 16*24*24 = 9216 ops.
        # Or use broadcasting.
        
        # u is (n*m,). Reshape to (n, m).
        u_reshaped = u.reshape(n, m)
        v_reshaped = v.reshape(m, p)
        w_reshaped = w.reshape(p, n)
        
        # We want term[i, j, k] = u_reshaped[i, j] * v_reshaped[j, k] * w_reshaped[k, i]
        # This is not a standard matmul.
        
        # Let's just loop over i, j, k for simplicity and speed (9216 iterations is fast).
        # Or use einsum on the reshaped versions?
        # u_reshaped[i, j] * v_reshaped[j, k] * w_reshaped[k, i]
        # This is a cyclic contraction.
        
        # Let's use a temporary 3D array for the term.
        term = np.einsum('ij,jk,ki->ijk', u_reshaped, v_reshaped, w_reshaped)
        
        # R is indexed by [i*m+j, j*p+k, k*n+i]
        # term is indexed by [i, j, k]
        # We need to map term[i, j, k] to R[i*m+j, j*p+k, k*n+i]
        
        # This mapping is tricky with simple slicing.
        # Let's just iterate.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    R[i*m+j, j*p+k, k*n+i] -= term[i, j, k]

    # Now we have R. We want to find u, v, w such that u[i*m+j] * v[j*p+k] * w[k*n+i] = R[i*m+j, j*p+k, k*n+i]
    # This is equivalent to finding rank-1 approximation of R in the "flattened" tensor space.
    # Let R_flat be a tensor of shape (n, m, p, n, m, p) ? No.
    # The indices are coupled.
    # Let's define the tensor S such that S[i, j, k] = R[i*m+j, j*p+k, k*n+i].
    # We want S[i, j, k] = u[i, j] * v[j, k] * w[k, i]
    # where u is (n, m), v is (m, p), w is (p, n).
    
    # This is a specific tensor decomposition problem.
    # We can use ALS on the 3D tensor S.
    
    S = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                S[i, j, k] = R[i*m+j, j*p+k, k*n+i]
                
    # Initialize u, v, w
    # Random init
    np.random.seed(42)
    
    success = False
    best_err = 1e9
    best_u, best_v, best_w = None, None, None
    
    # Multiple restarts
    max_restarts = 10
    max_iter = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        u = np.random.randn(n, m) * 0.1
        v = np.random.randn(m, p) * 0.1
        w = np.random.randn(p, n) * 0.1
        
        for it in range(max_iter):
            # Update u: u[i, j] = S[i, j, :] dot (v[j, :] * w[:, i]) / ||v[j, :] * w[:, i]||^2
            # Note: w[:, i] is the i-th column of w.
            
            # Let's compute the contraction for u.
            # u_new[i, j] = sum_k S[i, j, k] * v[j, k] * w[k, i]
            
            # Pre-compute v_w[i, j] = sum_k v[j, k] * w[k, i]? No, that's not right.
            # The term is sum_k S[i, j, k] * v[j, k] * w[k, i].
            
            # We can loop over i, j.
            u_new = np.zeros((n, m), dtype=np.float64)
            v_new = np.zeros((m, p), dtype=np.float64)
            w_new = np.zeros((p, n), dtype=np.float64)
            
            for i in range(n):
                for j in range(m):
                    # sum_k S[i, j, k] * v[j, k] * w[k, i]
                    # v[j, :] is (p,), w[:, i] is (p,)
                    # dot(S[i, j, :], v[j, :] * w[:, i])
                    val = np.dot(S[i, j, :], v[j, :] * w[:, i])
                    norm_sq = np.dot(v[j, :] * w[:, i], v[j, :] * w[:, i])
                    if norm_sq > 1e-12:
                        u_new[i, j] = val / norm_sq
                    else:
                        u_new[i, j] = u[i, j]
                        
            for j in range(m):
                for k in range(p):
                    # sum_i S[i, j, k] * u[i, j] * w[k, i]
                    # u[:, j] is (n,), w[k, :] is (n,)
                    val = np.dot(S[:, j, k], u[:, j] * w[k, :])
                    norm_sq = np.dot(u[:, j] * w[k, :], u[:, j] * w[k, :])
                    if norm_sq > 1e-12:
                        v_new[j, k] = val / norm_sq
                    else:
                        v_new[j, k] = v[j, k]
                        
            for k in range(p):
                for i in range(n):
                    # sum_j S[i, j, k] * u[i, j] * v[j, k]
                    # u[i, :] is (m,), v[:, k] is (m,)
                    val = np.dot(S[i, :, k], u[i, :] * v[:, k])
                    norm_sq = np.dot(u[i, :] * v[:, k], u[i, :] * v[:, k])
                    if norm_sq > 1e-12:
                        w_new[k, i] = val / norm_sq
                    else:
                        w_new[k, i] = w[k, i]
                        
            u, v, w = u_new, v_new, w_new
            
            # Check error
            # err = max |S[i, j, k] - u[i, j] v[j, k] w[k, i]|
            # Reconstruct
            recon = np.einsum('ij,jk,ki->ijk', u, v, w)
            err = np.max(np.abs(S - recon))
            
            if err < best_err:
                best_err = err
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
                
            if err < 1e-6:
                success = True
                break
                
        if success:
            break

    if success:
        # Construct the new U, V, W
        U_new = np.zeros((n*m, target_rank), dtype=np.float64)
        V_new = np.zeros((m*p, target_rank), dtype=np.float64)
        W_new = np.zeros((p*n, target_rank), dtype=np.float64)
        
        U_new[:, :target_rank-1] = U_base
        V_new[:, :target_rank-1] = V_base
        W_new[:, :target_rank-1] = W_base
        
        U_new[:, target_rank-1] = best_u.flatten()
        V_new[:, target_rank-1] = best_v.flatten()
        W_new[:, target_rank-1] = best_w.flatten()
        
        # Verify
        # We already know the error is < 1e-6, but let's be safe.
        # The verification in construct() will do the final check.
        return U_new.tolist(), V_new.tolist(), W_new.tolist(), target_rank

    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank
