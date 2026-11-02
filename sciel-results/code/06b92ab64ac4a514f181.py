<<<<<<< SEARCH
    def als_optimize(U, V, W, max_iters=200, tol=1e-12):
        """Alternating Least Squares optimization."""
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each column i, minimize sum_{b,c} (T[a,b,c] - U[a,i]*V[b,i]*W[c,i])^2 w.r.t U[:,i]
            # This is a linear least squares problem: U[:,i] * (V[:,i] ⊗ W[:,i])^T ≈ T_reshaped
            # More efficiently: for fixed i, we want U[:,i] to minimize ||T - V[:,i]*W[:,i]^T ⊗ U[:,i]||
            # Actually, let's do it properly with the tensor structure
            
            # For U update: fix V and W
            # T[a,b,c] = sum_i U[a,i] * V[b,i] * W[c,i]
            # For a fixed i, the contribution is U[a,i] * (V[b,i] * W[c,i])
            # So we need: U[:,i] * (V[:,i] ⊗ W[:,i])^T ≈ T_reshaped for each i
            # But this couples all i's. Let's use a different approach.
            
            # Alternative: use the fact that for each pair (b,c), 
            # T[:,b,c] = sum_i U[:,i] * V[b,i] * W[c,i]
            # = U @ (V[b,:] * W[c,:])
            
            # Let's build the system properly
            # For U: T[a,b,c] = sum_i U[a,i] * (V[b,i] * W[c,i])
            # Define M_U[b,c,i] = V[b,i] * W[c,i]
            # Then T[a,b,c] = sum_i U[a,i] * M_U[b,c,i]
            # Reshape: T_flat[a, (b,c)] = U[a,:] @ M_U[(b,c), :].T
            
            M_U = np.einsum('bi,ci->bci', V, W)  # (m*p, p*n, rank)
            M_U_flat = M_U.reshape(m*p*p*n, rank)  # (m*p*p*n, rank)
            T_flat = T.reshape(n*m, m*p*p*n)  # (n*m, m*p*p*n)
            
            # Solve: T_flat = U @ M_U_flat.T
            # U = T_flat @ M_U_flat.T @ (M_U_flat @ M_U_flat.T)^{-1}
            try:
                A = M_U_flat @ M_U_flat.T  # (rank, rank)
                A += 1e-10 * np.eye(rank)  # Regularization
                U_new = np.linalg.solve(A, M_U_flat.T @ T_flat.T).T
            except np.linalg.LinAlgError:
                U_new = U.copy()
            
            # Update V: fix U and W
            M_V = np.einsum('ai,ci->aci', U, W)  # (n*m, p*n, rank)
            M_V_flat = M_V.reshape(m*p, n*m*p*n)  # (m*p, n*m*p*n)
            T_flat_V = T.reshape(m*p, n*m*p*n).T  # (n*m*p*n, m*p)
            
            try:
                A = M_V_flat.T @ M_V_flat  # (rank, rank)
                A += 1e-11 * np.eye(rank)
                V_new = np.linalg.solve(A, M_V_flat.T @ T_flat_V.T).T
            except np.linalg.LinAlgError:
                V_new = V.copy()
            
            # Update W: fix U and V
            M_W = np.einsum('ai,bi->abi', U, V)  # (n*m, m*p, rank)
            M_W_flat = M_W.reshape(p*n, n*m*m*p)  # (p*n, n*m*m*p)
            T_flat_W = T.reshape(p*n, n*m*m*p).T  # (n*m*m*p, p*n)
            
            try:
                A = M_W_flat.T @ M_W_flat  # (rank, rank)
                A += 1e-11 * np.eye(rank)
                W_new = np.linalg.solve(A, M_W_flat.T @ T_flat_W.T).T
            except np.linalg.LinAlgError:
                W_new = W.copy()
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            if it % 10 == 0 or it == max_iters - 1:
                residual = compute_residual(U, V, W)
                if residual < tol:
                    break
        
        return U, V, W, compute_residual(U, V, W)
=======
    def als_optimize(U, V, W, max_iters=50, tol=1e-10):
        """Alternating Least Squares optimization with correct linear algebra."""
        T_flat_U = T.reshape(n*m, m*p*p*n)
        T_flat_V = T.reshape(m*p, n*m*p*n)
        T_flat_W = T.reshape(p*n, n*m*m*p)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            M_U = np.einsum('bi,ci->bci', V, W).reshape(m*p, rank)
            A_U = M_U.T @ M_U
            A_U += 1e-10 * np.eye(rank)
            B_U = T_flat_U @ M_U
            try:
                U = np.linalg.solve(A_U.T, B_U.T).T
            except np.linalg.LinAlgError:
                pass
            
            # Update V
            M_V = np.einsum('ai,ci->aci', U, W).reshape(n*p, rank)
            A_V = M_V.T @ M_V
            A_V += 1e-10 * np.eye(rank)
            B_V = T_flat_V @ M_V
            try:
                V = np.linalg.solve(A_V.T, B_V.T).T
            except np.linalg.LinAlgError:
                pass
            
            # Update W
            M_W = np.einsum('ai,bi->abi', U, V).reshape(n*m, rank)
            A_W = M_W.T @ M_W
            A_W += 1e-10 * np.eye(rank)
            B_W = T_flat_W @ M_W
            try:
                W = np.linalg.solve(A_W.T, B_W.T).T
            except np.linalg.LinAlgError:
                pass
            
            if it % 10 == 0 or it == max_iters - 1:
                residual = compute_residual(U, V, W)
                if residual < tol:
                    break
        
        return U, V, W, compute_residual(U, V, W)
>>>>>>> REPLACE

I will also modify the search loop to be less aggressive if it doesn't find a solution quickly, to ensure it returns the schoolbook solution. The current loop tries ranks 49 down to 30. This is computationally expensive. I'll limit the restarts and time per rank.

