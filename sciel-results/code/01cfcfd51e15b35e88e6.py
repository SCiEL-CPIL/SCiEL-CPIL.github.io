<<<<<<< SEARCH
    def _try_reduce_rank(U, V, W, R, max_iter=5000, tol=1e-6):
        """
        Attempt to reduce rank by eliminating one column at a time.
        For each candidate column r, solve for the minimum-norm update to
        the remaining columns that preserves the tensor decomposition.
        If the residual is small enough, accept the elimination.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        while R > 1 and (time.time() - t_start) < deadline:
            best_r = -1
            best_residual = np.inf
            best_U_new = None
            best_V_new = None
            best_W_new = None
            
            for r in range(R):
                if (time.time() - t_start) > deadline:
                    break
                # Try to eliminate column r
                # We need: sum_{s != r} U_s * V_s * W_s = U_r * V_r * W_r
                # Flatten: the tensor T has entries T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
                # We want to find delta_s for s != r such that:
                #   sum_s delta_s * (U[:,s]*V[:,s]*W[:,s]) = U[:,r]*V[:,r]*W[:,r]
                # This is a linear system in the 9*9=81 dimensional space
                # but we parameterize it as rank-1 updates.
                
                # Simpler approach: build the linear system
                # For each (i,j,k): sum_s delta_s * U[i,j,s]*V[j,k,s]*W[k,i,s] = U[i,j,r]*V[j,k,r]*W[k,i,r]
                
                A_mat = np.zeros((n*m*p, R-1))
                b_vec = np.zeros(n*m*p)
                
                col_idx = 0
                for s in range(R):
                    if s == r:
                        continue
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                A_mat[idx_u(i,j)*m*p + idx_v(j,k)*p + idx_w(k,i), col_idx] = \
                                    U[idx_u(i,j), s] * V[idx_v(j,k), s] * W[idx_w(k,i), s]
                    col_idx += 1
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            b_vec[idx_u(i,j)*m*p + idx_v(j,k)*p + idx_w(k,i)] = \
                                U[idx_u(i,j), r] * V[idx_v(j,k), r] * W[idx_w(k,i), r]
                
                # Solve least squares
                try:
                    delta, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=tol)
                except:
                    continue
                
                # Compute actual residual
                residual = np.linalg.norm(A_mat @ delta - b_vec)
                
                if residual < best_residual:
                    best_residual = residual
                    best_r = r
                    best_delta = delta.copy()
            
            if best_r < 0:
                break
            
            if best_residual < 1e-4:
                # Apply the elimination
                U_new = np.zeros((n*m, R-1))
                V_new = np.zeros((m*p, R-1))
                W_new = np.zeros((p*n, R-1))
                
                col_idx = 0
                for s in range(R):
                    if s == best_r:
                        continue
                    U_new[:, col_idx] = U[:, s] + best_delta[col_idx] * 0  # placeholder
                    V_new[:, col_idx] = V[:, s]
                    W_new[:, col_idx] = W[:, s]
                    col_idx += 1
                
                # Actually, the delta modifies the coefficients, not the matrices directly.
                # The correct approach: the new decomposition is
                # T = sum_{s != r} (U_s + delta_s * something) ... 
                # This is more complex. Let's use a different formulation.
                
                # Alternative: express the rank-r term as a combination of others
                # T_r = U_r ⊗ V_r ⊗ W_r = sum_{s != r} c_s * U_s ⊗ V_s ⊗ W_s
                # This only works if T_r is in the span of the other terms, which
                # requires the tensor to be linearly dependent.
                
                # For now, skip this and try a simpler approach
                pass
            else:
                break
        
        return U, V, W, R
=======
    def _try_reduce_rank(U, V, W, R, tol=1e-6):
        """
        Attempt to reduce rank by checking if any rank-1 term is in the
        span of the others. If so, eliminate it.
        
        A term r can be eliminated if the tensor T_r = U_r ⊗ V_r ⊗ W_r
        can be written as sum_{s != r} c_s * T_s, i.e., if T_r is in the
        linear span of {T_s : s != r} in R^{27}.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        while R > 1 and (time.time() - t_start) < deadline:
            best_r = -1
            best_residual = np.inf
            
            for r in range(R):
                if (time.time() - t_start) > deadline:
                    break
                # Build tensor for term r
                T_r = np.zeros(n*m*p)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_r[idx_u(i,j)*m*p + idx_v(j,k)*p + idx_w(k,i)] = \
                                U[idx_u(i,j), r] * V[idx_v(j,k), r] * W[idx_w(k,i), r]
                
                # Build matrix of other terms
                A_mat = np.zeros((n*m*p, R-1))
                col_idx = 0
                for s in range(R):
                    if s == r:
                        continue
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                A_mat[idx_u(i,j)*m*p + idx_v(j,k)*p + idx_w(k,i), col_idx] = \
                                    U[idx_u(i,j), s] * V[idx_v(j,k), s] * W[idx_w(k,i), s]
                    col_idx += 1
                
                # Check if T_r is in span of other terms
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(A_mat, T_r, rcond=tol)
                except:
                    continue
                
                residual = np.linalg.norm(A_mat @ coeffs - T_r)
                
                if residual < best_residual:
                    best_residual = residual
                    best_r = r
            
            if best_r < 0:
                break
            
            if best_residual < 1e-8:
                # Eliminate column best_r
                U = np.delete(U, best_r, axis=1)
                V = np.delete(V, best_r, axis=1)
                W = np.delete(W, best_r, axis=1)
                R -= 1
            else:
                break
        
        return U, V, W, R
>>>>>>> REPLACE

Now let me also fix the `_build_rank22` function which has a bug (it sets R=22 but then reassigns R=26):

