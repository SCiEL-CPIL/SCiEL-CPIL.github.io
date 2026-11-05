import numpy as np

def construct():
    n, m, p = 3, 4, 4
    
    # Try ranks from low to high
    for rank in [32, 36, 40, 44, 48]:
        U, V, W = _try_decompose(n, m, p, rank)
        if U is not None:
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    return {"U": U, "V": V, "W": W}

def _try_decompose(n, m, p, rank, max_iters=2000, tol=1e-10):
    np.random.seed(42)
    
    # Initialize
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    for it in range(max_iters):
        # Fix U, V, solve for W
        # For each (k, i), W[k*n+i, :] = sum over j of T[i,j,k] / (U[i*m+j,:] * V[j*p+k,:])
        # This is a least squares problem
        # W[k*n+i, c] = sum_j T[i,j,k] * U[i*m+j, c] * V[j*p+k, c] / ... 
        # Actually, it's: sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*V[j*p+k,c]) = T[i,j,k] for each j
        # No, the equation is: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For fixed U,V, this is linear in W: for each (i,j,k), sum_c (U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = T[i,j,k]
        # But T[i,j,k] = 1 for all i,j,k in the matmul tensor.
        
        # Actually, the matmul tensor has T[i,j,k] = 1 for all valid i,j,k.
        # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
        
        # Fix U, V: for each (k,i), we have equations for each j:
        # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
        # This is m equations (one per j) for p*n unknowns (W rows).
        # Actually, for fixed (k,i), the unknowns are W[k*n+i, 0..rank-1].
        # And we have m equations (for j=0..m-1).
        
        # So for each (k,i), solve: A_{j,c} * W_c = 1 for j=0..m-1
        # where A_{j,c} = U[i*m+j,c] * V[j*p+k,c]
        
        # This is an underdetermined system (m equations, rank unknowns). Use least squares.
        
        new_W = np.zeros((p * n, rank))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                # Solve A @ W_vec = ones(m)
                W_vec, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                new_W[k*n+i, :] = W_vec
        W = new_W
        
        # Fix U, W, solve for V
        new_V = np.zeros((m * p, rank))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                V_vec, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                new_V[j*p+k, :] = V_vec
        V = new_V
        
        # Fix V, W, solve for U
        new_U = np.zeros((n * m, rank))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                U_vec, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                new_U[i*m+j, :] = U_vec
        U = new_U
        
        # Check convergence
        if it % 50 == 0:
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1)**2
            if err < tol:
                # Convert to list of lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return U_list, V_list, W_list
    
    # Check final error
    err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                err += (val - 1)**2
    if err < 1e-6:
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return U_list, V_list, W_list
    
    return None, None, None
