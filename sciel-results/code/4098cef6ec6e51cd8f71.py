import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (12, rank)
    # W: (p*n, rank) = (12, rank)
    # Constraint: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
    
    best_rank = n * m * p  # fallback to schoolbook
    best_solution = None
    
    # Try ranks from low to high
    for rank in range(16, n * m * p + 1):
        # Try multiple random initializations
        found = False
        for trial in range(50):
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Alternating least squares
            converged = False
            for it in range(200):
                # Update W: fix U, V
                for i in range(n):
                    for k in range(p):
                        # For fixed (i,k), equations over j:
                        # sum_s W[k*n+i, s] * U[i*m+j, s] * V[j*p+k, s] = 1 for j=0..m-1
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.ones(m)
                        # Solve A @ W[k*n+i, :] = b
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        W[k * n + i, :] = sol
                
                # Update U: fix V, W
                for i in range(n):
                    for j in range(m):
                        # For fixed (i,j), equations over k:
                        # sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1 for k=0..p-1
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b = np.ones(p)
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        U[i * m + j, :] = sol
                
                # Update V: fix U, W
                for j in range(m):
                    for k in range(p):
                        # For fixed (j,k), equations over i:
                        # sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1 for i=0..n-1
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.ones(n)
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        V[j * p + k, :] = sol
                
                # Check residual
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-6:
                    converged = True
                    break
            
            if converged:
                # Round to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exact decomposition with rounded values
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = rank
                    best_solution = (U_r, V_r, W_r)
                    found = True
                    break
        
        if found:
            break
    
    if best_solution is not None:
        U, V, W = best_solution
        # Convert to lists of lists
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
