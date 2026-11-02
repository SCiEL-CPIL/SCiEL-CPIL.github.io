<<<<<<< SEARCH
    def try_rank_fixed_VW(R, n, m, p, num_trials=200, time_budget=8.0):
        """
        Try to find a rank-R decomposition by fixing V and W randomly,
        then solving for U via least squares. Verify the result.
        """
        start = time.time()
        for trial in range(num_trials):
            if time.time() - start > time_budget:
                break
            
            # Random V and W
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Build the linear system for U
            # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # For fixed (i,j): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
            # This gives a system: A[i,j] @ U[i*m+j, :] = ones(p)
            # where A[i,j][k, r] = V[j*p+k, r] * W[k*n+i, r]
            
            # Build A as (n*m*p, R) and b as (n*m*p,)
            A = np.zeros((n * m * p, R))
            b = np.ones(n * m * p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        A[idx, :] = V[j * p + k, :] * W[k * n + i, :]
                        idx += 1
            
            # Solve least squares: min ||A @ U_flat - b||^2
            # U_flat is (n*m*R,)
            try:
                U_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            U = U_flat.reshape(n * m, R)
            
            # Verify
            if _verify_fast(U, V, W, n, m, p):
                return U, V, W
        
        return None
=======
    def try_rank_fixed_VW(R, n, m, p, num_trials=200, time_budget=8.0):
        """
        Try to find a rank-R decomposition by fixing V and W randomly,
        then solving for U rows via least squares.
        """
        start = time.time()
        for trial in range(num_trials):
            if time.time() - start > time_budget:
                break
            
            # Random V and W
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            U = np.zeros((n * m, R))
            
            valid = True
            for i in range(n):
                for j in range(m):
                    # Construct A_ij (p x R)
                    # A_ij[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A_ij = np.zeros((p, R))
                    for k in range(p):
                        A_ij[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    
                    b = np.ones(p)
                    try:
                        x, residuals, rank_A, sv = np.linalg.lstsq(A_ij, b, rcond=None)
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    # Check if solution is exact
                    if residuals is not None and len(residuals) > 0:
                        if residuals[0] > 1e-9:
                            valid = False
                            break
                    else:
                        # If underdetermined, lstsq might not return residuals or they are 0
                        # Check residual manually
                        res = np.linalg.norm(A_ij @ x - b)
                        if res > 1e-9:
                            valid = False
                            break
                    
                    U[i * m + j, :] = x
            
            if valid:
                # Final verification just in case
                if _verify_fast(U, V, W, n, m, p):
                    return U, V, W
        
        return None
>>>>>>> REPLACE

