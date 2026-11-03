<<<<<<< SEARCH
    # 2. Attempt to reduce rank to 35 using local optimization
    # We try to zero out one column by adjusting the others
    
    def try_reduce_rank(U, V, W, target_rank, max_iters=20):
        if target_rank >= U.shape[1]:
            return None
        
        U_curr = U[:, :target_rank].copy()
        V_curr = V[:, :target_rank].copy()
        W_curr = W[:, :target_rank].copy()
        
        # Simple alternating minimization to see if we can fit into target_rank
        # This is a heuristic; exact integer solution is preferred
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
                
            # Update U
            C = np.zeros((target_rank, m*p * p*n))
            for r in range(target_rank):
                C[r, :] = np.kron(V_curr[:, r], W_curr[:, r])
            T = np.zeros((n*m, m*p * p*n))
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        row_a, col_a = divmod(i, m)
                        col_a2, row_b = divmod(j, p)
                        col_b, row_a2 = divmod(k, n)
                        if col_a == col_a2 and row_a == row_a2:
                            T[i, j*p*n + k] = 1.0
            # T_flat is (n*m, m*p*p*n)
            # C.T is (m*p*p*n, target_rank)
            # We want U_curr such that U_curr @ C.T = T_flat.T
            # This is a least squares problem
            try:
                U_new, _, _, _ = np.linalg.lstsq(C.T, T.T, rcond=None)
                U_curr = U_new.T
            except:
                return None

            # Update V
            D = np.zeros((target_rank, n*m * p*n))
            for r in range(target_rank):
                D[r, :] = np.kron(U_curr[:, r], W_curr[:, r])
            T2 = np.zeros((m*p, n*m * p*n))
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        row_a, col_a = divmod(i, m)
                        col_a2, row_b = divmod(j, p)
                        col_b, row_a2 = divmod(k, n)
                        if col_a == col_a2 and row_a == row_a2:
                            T2[j, i*p*n + k] = 1.0
            try:
                V_new, _, _, _ = np.linalg.lstsq(D.T, T2.T, rcond=None)
                V_curr = V_new.T
            except:
                return None

            # Update W
            E = np.zeros((target_rank, n*m * m*p))
            for r in range(target_rank):
                E[r, :] = np.kron(U_curr[:, r], V_curr[:, r])
            T3 = np.zeros((p*n, n*m * m*p))
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        row_a, col_a = divmod(i, m)
                        col_a2, row_b = divmod(j, p)
                        col_b, row_a2 = divmod(k, n)
                        if col_a == col_a2 and row_a == row_a2:
                            T3[k, i*m*p + j] = 1.0
            try:
                W_new, _, _, _ = np.linalg.lstsq(E.T, T3.T, rcond=None)
                W_curr = W_new.T
            except:
                return None

        # Check if the solution is close to integers
        all_entries = np.concatenate([U_curr.flatten(), V_curr.flatten(), W_curr.flatten()])
        rounded = np.round(all_entries)
        round_err = np.sum((all_entries - rounded)**2)
        
        if round_err < 1e-4:
            U_r = np.round(U_curr)
            V_r = np.round(V_curr)
            W_r = np.round(W_curr)
            if verify_np(U_r, V_r, W_r):
                return U_r, V_r, W_r
        return None

    # Try rank 35
    if time.time() < deadline:
        result = try_reduce_rank(U, V, W, 35)
        if result:
            U, V, W = result
            return {"U": U.astype(int).tolist(), "V": V.astype(int).tolist(), "W": W.astype(int).tolist()}

    # Return the rank-36 solution
    return {"U": U.astype(int).tolist(), "V": V.astype(int).tolist(), "W": W.astype(int).tolist()}
=======
    # 2. Attempt to reduce rank using tensor slicing + alternating least squares
    
    def build_tensor_slices():
        """Build the target tensor as a 3D array T[i,j,k]"""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n*m):
            row_a, col_a = divmod(i, m)
            for j in range(m*p):
                col_a2, row_b = divmod(j, p)
                if col_a == col_a2:
                    for k in range(p*n):
                        col_b, row_a2 = divmod(k, n)
                        if row_a == row_a2 and col_b == row_b:
                            T[i, j, k] = 1.0
        return T

    def try_reduce_rank_als(U_init, V_init, W_init, target_rank, max_iters=50, tol=1e-6):
        """Alternating least squares to find a rank-target_rank decomposition"""
        if target_rank >= U_init.shape[1]:
            return None
        
        T = build_tensor_slices()
        T_flat_i = T.reshape(n*m, -1)  # (n*m, m*p*p*n)
        T_flat_j = T.transpose(1, 0, 2).reshape(m*p, -1)  # (m*p, n*m*p*n)
        T_flat_k = T.transpose(2, 0, 1).reshape(p*n, -1)  # (p*n, n*m*m*p)
        
        # Initialize from the rank-36 solution by truncating
        U_curr = U_init[:, :target_rank].astype(float).copy()
        V_curr = V_init[:, :target_rank].astype(float).copy()
        W_curr = W_init[:, :target_rank].astype(float).copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: minimize ||T_flat_i - U @ kron(V,W)||_F
            # T_flat_i[i, j*p*n+k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # For fixed V, W: U[i,:] solves U[i,:] @ (kron(V[:,r], W[:,r]).T) = T_flat_i[i,:]
            # Build matrix M where M[j*p*n+k, r] = V[j,r] * W[k,r]
            M = np.zeros((m*p*p*n, target_rank))
            for r in range(target_rank):
                M[:, r] = np.kron(V_curr[:, r], W_curr[:, r])
            try:
                U_new, _, _, _ = np.linalg.lstsq(M.T, T_flat_i.T, rcond=None)
                U_curr = U_new.T
            except:
                return None
            
            # Update V
            M2 = np.zeros((n*m*p*n, target_rank))
            for r in range(target_rank):
                M2[:, r] = np.kron(U_curr[:, r], W_curr[:, r])
            try:
                V_new, _, _, _ = np.linalg.lstsq(M2.T, T_flat_j.T, rcond=None)
                V_curr = V_new.T
            except:
                return None
            
            # Update W
            M3 = np.zeros((n*m*m*p, target_rank))
            for r in range(target_rank):
                M3[:, r] = np.kron(U_curr[:, r], V_curr[:, r])
            try:
                W_new, _, _, _ = np.linalg.lstsq(M3.T, T_flat_k.T, rcond=None)
                W_curr = W_new.T
            except:
                return None
            
            # Check convergence
            if it > 0:
                diff = np.max(np.abs(U_curr - U_prev)) + np.max(np.abs(V_curr - V_prev)) + np.max(np.abs(W_curr - W_prev))
                if diff < tol:
                    break
            U_prev, V_prev, W_prev = U_curr.copy(), V_curr.copy(), W_curr.copy()
        
        # Try rounding to integers or half-integers
        for scale in [1, 2]:
            U_r = np.round(U_curr * scale) / scale
            V_r = np.round(V_curr * scale) / scale
            W_r = np.round(W_curr * scale) / scale
            
            # Verify
            if verify_np(U_r, V_r, W_r):
                # Convert to integer representation if using scale=2
                if scale == 2:
                    U_r = (U_r * 2).astype(int)
                    V_r = (V_r * 2).astype(int)
                    W_r = (W_r * 2).astype(int)
                else:
                    U_r = U_r.astype(int)
                    V_r = V_r.astype(int)
                    W_r = W_r.astype(int)
                return U_r, V_r, W_r
        
        return None

    # Try rank 35 with multiple initializations
    best_result = None
    for trial in range(3):
        if time.time() > deadline - 10:
            break
        
        if trial == 0:
            # Use the rank-36 solution truncated
            U_try, V_try, W_try = try_reduce_rank_als(U, V, W, 35, max_iters=30)
        else:
            # Random perturbation
            rng = np.random.RandomState(42 + trial)
            U_rand = U[:, :35].copy() + rng.randn(n*m, 35) * 0.1
            V_rand = V[:, :35].copy() + rng.randn(m*p, 35) * 0.1
            W_rand = W[:, :35].copy() + rng.randn(p*n, 35) * 0.1
            U_try, V_try, W_try = try_reduce_rank_als(U_rand, V_rand, W_rand, 35, max_iters=30)
        
        if U_try is not None:
            best_result = (U_try, V_try, W_try)
            break
    
    if best_result:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Return the rank-36 solution
    return {"U": U.astype(int).tolist(), "V": V.astype(int).tolist(), "W": W.astype(int).tolist()}
>>>>>>> REPLACE
