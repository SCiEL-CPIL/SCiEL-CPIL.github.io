import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Try to find low-rank decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for rank in range(24, 64):
        for attempt in range(50):
            np.random.seed(attempt * 1000 + rank)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            converged = False
            for iteration in range(200):
                # Solve for U given V, W
                for i in range(n):
                    for j in range(m):
                        # For each k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # Build matrix A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve A @ U[i*m+j, :]^T = ones(p)
                        # A is p x rank, we want x (rank x 1) such that A @ x = ones(p)
                        # This is: A @ x = b, where b = ones(p)
                        # Use least squares
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i*m+j, :] = sol
                
                # Solve for V given U, W
                for j in range(m):
                    for k in range(p):
                        # For each i: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # Build matrix A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j*p+k, :] = sol
                
                # Solve for W given U, V
                for k in range(p):
                    for i in range(n):
                        # For each j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # Build matrix A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k*n+i, :] = sol
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
