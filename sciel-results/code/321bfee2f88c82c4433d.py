def _als_reduce_fast(T, rank_target, max_iters=200, n_restarts=10, deadline=None):
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n
    
    best_result = None
    best_err = float('inf')
    
    for restart in range(n_restarts):
        if deadline and time.time() > deadline - 5:
            break
        
        U = np.random.randn(Nm, rank_target)
        V = np.random.randn(Mp, rank_target)
        W = np.random.randn(Pn, rank_target)
        
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if deadline and time.time() > deadline - 5:
                break
            
            # Update U
            # H[i, j, k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Reshape V to (m, p, rank)
            V_reshaped = V.reshape(m, p, rank_target)
            # Reshape W to (p, n, rank)
            W_reshaped = W.reshape(p, n, rank_target)
            
            # We need for each i, j:
            # A[i, j, k, r] = V_reshaped[j, k, r] * W_reshaped[k, i, r]
            # This is a broadcast product.
            # V_reshaped: (m, p, rank)
            # W_reshaped: (p, n, rank)
            # We want A of shape (n, m, p, rank)
            # A[i, j, k, r] = V_reshaped[j, k, r] * W_reshaped[k, i, r]
            # Note indices: i from W (2nd dim), j from V (1st dim), k from both (1st/2nd dim).
            
            # Let's construct A.
            # V_reshaped[:, :, :] -> (1, m, p, rank)
            # W_reshaped[:, :, :] -> (p, n, rank) -> (n, 1, p, rank) ? No.
            # W_reshaped[k, i, r]. We need i, j, k, r.
            # W_reshaped.T is (n, p, rank).
            # So W_part = W_reshaped.T -> (n, p, rank).
            # V_part = V_reshaped -> (m, p, rank).
            # A[i, j, k, r] = V_part[j, k, r] * W_part[i, k, r].
            
            V_part = V_reshaped  # (m, p, rank)
            W_part = W_reshaped.T # (n, p, rank)
            
            # Broadcast: V_part[None, :, :, :] * W_part[:, None, :, :]
            # Shape: (n, m, p, rank)
            A = V_part[None, :, :, :] * W_part[:, None, :, :]
            
            # Reshape A to (n*m, p, rank)
            A_flat = A.reshape(Nm, p, rank_target)
            T_flat = T.reshape(Nm, p)
            
            # Solve A_flat[i] @ U_flat[i].T = T_flat[i].T
            # for i in range(Nm)
            U_new = np.zeros((Nm, rank_target))
            for idx in range(Nm):
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_flat[idx], T_flat[idx].T, rcond=None)
                    U_new[idx] = sol
                except:
                    pass
            U = U_new
            
            # Update V
            # For fixed j, k:
            # sum_r U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r] = T[i,j,k]
            # A[i, j, r] = U[i*m+j, r] * W[k*n+i, r]
            # Reshape U to (n, m, rank)
            U_reshaped = U.reshape(n, m, rank_target)
            # Reshape W to (p, n, rank)
            W_reshaped = W.reshape(p, n, rank_target)
            
            # We need A of shape (m, p, n*m, rank)? No.
            # For each j, k, we solve for V[j*p+k].
            # The equation is: sum_r (U_reshaped[i, j, r] * W_reshaped[k, i, r]) * V[j*p+k, r] = T[i, j, k]
            # Let B[i, j, k, r] = U_reshaped[i, j, r] * W_reshaped[k, i, r]
            # Shape of B: (n, m, p, rank)
            # We want to solve for each (j, k):
            # sum_r B[i, j, k, r] * V[j*p+k, r] = T[i, j, k]
            # This is a linear system in V[j*p+k, :].
            # The matrix is B[:, j, k, :] of shape (n, rank).
            # The RHS is T[:, j, k] of shape (n,).
            
            # Construct B:
            # U_reshaped: (n, m, rank) -> (n, m, 1, rank)
            # W_reshaped: (p, n, rank) -> (1, 1, p, rank)? No.
            # W_reshaped[k, i, r]. We need i, j, k, r.
            # W_reshaped.T is (n, p, rank).
            # W_part_V = W_reshaped.T -> (n, p, rank).
            # U_part_V = U_reshaped -> (n, m, rank).
            # B[i, j, k, r] = U_part_V[i, j, r] * W_part_V[i, k, r].
            # Broadcast: U_part_V[:, :, None, :] * W_part_V[:, None, :, :]
            # Shape: (n, m, p, rank)
            B = U_reshaped[:, :, None, :] * W_reshaped.T[:, None, :, :]
            
            V_new = np.zeros((Mp, rank_target))
            for j in range(m):
                for k in range(p):
                    # B[:, j, k, :] is (n, rank)
                    # T[:, j, k] is (n,)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(B[:, j, k, :], T[:, j, k], rcond=None)
                        V_new[j * p + k] = sol
                    except:
                        pass
            V = V_new
            
            # Update W
            # For fixed k, i:
            # sum_r U[i*m+j, r] * V[j*p+kk, r] * W[k*n+i, r] = T[i, j, kk]
            # Let C[j, kk, r] = U[i*m+j, r] * V[j*p+kk, r]
            # Reshape U to (n, m, rank)
            U_reshaped = U.reshape(n, m, rank_target)
            # Reshape V to (m, p, rank)
            V_reshaped = V.reshape(m, p, rank_target)
            
            # C[i, j, kk, r] = U_reshaped[i, j, r] * V_reshaped[j, kk, r]
            # Shape: (n, m, p, rank)
            C = U_reshaped[:, :, None, :] * V_reshaped[None, :, :, :]
            
            W_new = np.zeros((Pn, rank_target))
            for k in range(p):
                for i in range(n):
                    # We need to solve for W[k*n+i, :]
                    # sum_r C[i, j, k, r] * W[k*n+i, r] = T[i, j, k]
                    # Matrix: C[i, :, k, :] of shape (m, rank)
                    # RHS: T[i, :, k] of shape (m,)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(C[i, :, k, :], T[i, :, k], rcond=None)
                        W_new[k * n + i] = sol
                    except:
                        pass
            W = W_new
            
            # Compute error
            # T_recon[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Use einsum
            # U: (n, m, rank), V: (m, p, rank), W: (p, n, rank)
            T_recon = np.einsum('ijr,jkr,kir->ijk', U.reshape(n, m, rank_target), V.reshape(m, p, rank_target), W.reshape(p, n, rank_target))
            err = np.linalg.norm(T_recon - T)
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                break
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
        
        if best_err < 1e-9:
            U, V, W = best_result
            # Round to half-integers
            U_list = [[round(U[i, r] * 2) / 2 for r in range(rank_target)] for i in range(Nm)]
            V_list = [[round(V[i, r] * 2) / 2 for r in range(rank_target)] for i in range(Mp)]
            W_list = [[round(W[i, r] * 2) / 2 for r in range(rank_target)] for i in range(Pn)]
            
            if _verify(U_list, V_list, W_list, T):
                return {"U": U_list, "V": V_list, "W": W_list}
            
            U_list = [[round(U[i, r]) for r in range(rank_target)] for i in range(Nm)]
            V_list = [[round(V[i, r]) for r in range(rank_target)] for i in range(Mp)]
            W_list = [[round(W[i, r]) for r in range(rank_target)] for i in range(Pn)]
            if _verify(U_list, V_list, W_list, T):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
