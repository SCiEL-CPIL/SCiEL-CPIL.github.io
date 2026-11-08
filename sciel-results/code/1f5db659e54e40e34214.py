# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition (rank = n*m*p = 50)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.float64)
    V_school = np.zeros((mp, rank_school), dtype=np.float64)
    W_school = np.zeros((pn, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s kill

    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    def verify_decomposition(U, V, W, T):
        """Check if U, V, W exactly decompose T."""
        # T_comp[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_comp, T, atol=1e-8)

    def try_als_rank(R, max_time_per_rank=10.0):
        """Attempt to find a decomposition of rank R using ALS."""
        nonlocal best_result, best_rank
        
        # We will run multiple restarts
        max_restarts = 50
        
        for attempt in range(max_restarts):
            if time.time() > start_time + max_time_per_rank * (R - 29) / 20: # Heuristic time allocation
                break
            if time.time() > deadline - 5.0:
                return False

            np.random.seed(42 + R * 1000 + attempt)
            
            # Initialize U, V, W randomly
            # Scaling is important. 
            # U, V, W should be roughly of magnitude 1/sqrt(R) to keep T_comp ~ 1
            U = np.random.randn(nm, R) / np.sqrt(R)
            V = np.random.randn(mp, R) / np.sqrt(R)
            W = np.random.randn(pn, R) / np.sqrt(R)
            
            # ALS Iterations
            for it in range(100):
                if time.time() > deadline - 5.0:
                    return False
                
                # Update U
                # U[:, r] = argmin_U sum_i (T[i,:,:] * (V[r] outer W[r])) - U[i,r] ...
                # Actually, for each r, U[:, r] solves:
                # sum_r U[i,r] V[j,r] W[k,r] = T[i,j,k]
                # Fix V, W. For a fixed r, the term involving U[:, r] is:
                # sum_i U[i,r] V[j,r] W[k,r]
                # This is a linear system for U[:, r] if we view it as a vector?
                # No, it's coupled across i.
                # Let's use the standard ALS update for tensor CP decomposition.
                # U[:, r] = (V[r] outer W[r])^T vec(T) ... no.
                
                # Correct ALS step:
                # For each r:
                # U[:, r] is updated by minimizing sum_{i,j,k} (T[i,j,k] - sum_{r'} U[i,r'] V[j,r'] W[k,r'])^2
                # w.r.t U[i,r].
                # This is a linear least squares problem.
                # Let A_r be the matrix where A_r[i, (j,k)] = V[j,r] * W[k,r].
                # Then U[:, r] = argmin_U sum_i (T_vec[i] - sum_{r'} U[i,r'] ... )
                # Wait, the tensor is 3-way.
                # For fixed r, U[:, r] appears in the sum as:
                # sum_i U[i,r] * (V[j,r] W[k,r])
                # This can be seen as:
                # vec(T) = sum_r U[:,r] (V[:,r] outer W[:,r])^T
                # This is a linear system:
                # M * vec([U, V, W]) = vec(T)
                # But we are doing ALS, so we fix V, W.
                # Then for each r, we want to update U[:, r].
                # The contribution of U[:, r] to T is:
                # T_r = U[:, r] (V[:, r] outer W[:, r])^T
                # We want to minimize || T - sum_{r'} U[:, r'] (V[:, r'] outer W[:, r'])^T ||^2
                # This is not simply decoupled for U[:, r] because the other terms U[:, r'] are fixed.
                # Let R_other = T - sum_{r' != r} U[:, r'] (V[:, r'] outer W[:, r'])^T
                # Then we want to minimize || R_other - U[:, r] (V[:, r] outer W[:, r])^T ||^2
                # This is a linear least squares problem for U[:, r].
                # Let B_r = (V[:, r] outer W[:, r])^T reshaped to (nm, mp*pn)?
                # No, V[:, r] outer W[:, r] is (mp, pn).
                # The term is U[:, r] (V[:, r] outer W[:, r])^T.
                # This is an outer product of U[:, r] (nm) and V[:, r] outer W[:, r] (mp, pn)?
                # No, the sum is over r.
                # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
                # For fixed r, the term is U[i,r] V[j,r] W[k,r].
                # This is not a standard matrix product.
                
                # Let's use the standard trick:
                # Fix V, W.
                # For each i, T[i, :, :] = sum_r U[i,r] (V[:, r] outer W[:, r])
                # So, T[i, :, :] is a linear combination of the matrices M_r = V[:, r] outer W[:, r].
                # We want to find U[i,r] such that T[i] approx sum_r U[i,r] M_r.
                # This is a linear least squares problem for U[i, :] for each i.
                # Let M_mat be a matrix of shape (mp*pn, R) where column r is vec(M_r).
                # Let T_mat be a matrix of shape (nm, mp*pn) where row i is vec(T[i]).
                # Then T_mat = U * M_mat^T.
                # We want to solve for U (nm x R).
                # U = T_mat * pinv(M_mat^T)
                
                M_mat = np.zeros((mp * pn, R))
                for r in range(R):
                    M_mat[:, r] = (V[:, r] * W[:, r][:, None]).flatten()
                
                T_mat = T.reshape(nm, mp * pn)
                
                # Solve U = T_mat @ M_mat^T @ inv(M_mat @ M_mat^T)
                # Add regularization to avoid singularity
                MMT = M_mat.T @ M_mat + 1e-6 * np.eye(R)
                U_new = T_mat @ M_mat @ np.linalg.solve(MMT, np.eye(R))
                
                # Update V
                # Fix U, W.
                # T[:, j, :] = sum_r V[j,r] (U[:, r] outer W[:, r])
                # N_mat shape (nm*pn, R) where column r is vec(U[:, r] outer W[:, r])
                N_mat = np.zeros((nm * pn, R))
                for r in range(R):
                    N_mat[:, r] = (U[:, r] * W[:, r][:, None]).flatten()
                
                T_mat_V = T.reshape(mp, nm * pn).T # (nm*pn, mp)
                # We want T_mat_V = N_mat @ V.T
                # V.T = N_mat^T @ T_mat_V ... no
                # T_mat_V (nm*pn, mp) = N_mat (nm*pn, R) @ V.T (R, mp)
                # So V.T = pinv(N_mat) @ T_mat_V
                NNT = N_mat.T @ N_mat + 1e-6 * np.eye(R)
                V_new = np.linalg.solve(NNT, N_mat.T @ T_mat_V).T
                
                # Update W
                # Fix U, V.
                # T[:, :, k] = sum_r W[k,r] (U[:, r] outer V[:, r])
                P_mat = np.zeros((nm * mp, R))
                for r in range(R):
                    P_mat[:, r] = (U[:, r] * V[:, r][:, None]).flatten()
                
                T_mat_W = T.reshape(pn, nm * mp).T # (nm*mp, pn)
                # T_mat_W = P_mat @ W.T
                PPT = P_mat.T @ P_mat + 1e-6 * np.eye(R)
                W_new = np.linalg.solve(PPT, P_mat.T @ T_mat_W).T
                
                # Check convergence
                U = U_new
                V = V_new
                W = W_new
                
                # Compute error
                T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.sum((T_comp - T) ** 2)
                if err < 1e-10:
                    break
            
            # After ALS, try rounding
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            if verify_decomposition(U_int, V_int, W_int, T):
                best_rank = R
                best_result = {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
                return True
            
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            
            if verify_decomposition(U_half, V_half, W_half, T):
                best_rank = R
                best_result = {
                    "U": U_half.tolist(),
                    "V": V_half.tolist(),
                    "W": W_half.tolist()
                }
                return True

        return False

    # Try candidate ranks in increasing order
    # Known lower bound is ~30. Let's try 30 to 49.
    candidate_ranks = list(range(30, 50))
    
    for R in candidate_ranks:
        if time.time() > deadline - 5.0:
            break
        
        success = try_als_rank(R)
        if success:
            break

    return best_result
# EVOLVE-BLOCK-END
