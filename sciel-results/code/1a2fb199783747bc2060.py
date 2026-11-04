"""Matrix multiplication 2,5,5 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    """Construct rank-minimal tensor decomposition for 2x5 by 5x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 2, 5, 5
    
    # Constraint: for all i in [0,2), j in [0,5), k in [0,5):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # ALS approach for the coupled index structure.
    
    def _als_coupled(U, V, W, rank, max_iters=60, tol=1e-11):
        """ALS for coupled matmul tensor structure."""
        for it in range(max_iters):
            # Update U: for each (i,j), solve linear system for U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    # Variables: U[i*m+j, 0..rank-1]
                    # Equations: for each k in range(p):
                    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    V_rows = V[j*p + np.arange(p), :]  # (p, rank)
                    W_rows = W[np.arange(p)*n + i, :]  # (p, rank)
                    A = V_rows * W_rows  # (p, rank)
                    b = np.ones(p)
                    U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            for j in range(m):
                for k in range(p):
                    # Equations: for each i in range(n):
                    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    U_rows = U[np.arange(n)*m + j, :]  # (n, rank)
                    W_rows = W[np.arange(n)*n + k, :]  # (n, rank)  # W[k*n+i] for i in range(n)
                    A = U_rows * W_rows  # (n, rank)
                    b = np.ones(n)
                    V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            for k in range(p):
                for i in range(n):
                    # Equations: for each j in range(m):
                    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    U_rows = U[i*m + np.arange(m), :]  # (m, rank)
                    V_rows = V[np.arange(m)*p + k, :]  # (m, rank)
                    A = U_rows * V_rows  # (m, rank)
                    b = np.ones(m)
                    W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Compute error
            total_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        total_err += (val - 1.0)**2
            err = np.sqrt(total_err)
            if err < tol:
                break
        
        return U, V, W, err
    
    # Try ranks from 45 down to 40
    for rank in [45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(rank * 137 + 42)
        best_err = np.inf
        best_U, best_V, best_W = None, None, None
        
        for restart in range(8):
            if time.time() > deadline:
                break
            
            U = rng.randn(n*m, rank) * 0.5
            V = rng.randn(m*p, rank) * 0.5
            W = rng.randn(p*n, rank) * 0.5
            
            U, V, W, err = _als_coupled(U, V, W, rank, max_iters=50, tol=1e-11)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-10 and best_U is not None:
            U_list = [[float(best_U[i, r]) for r in range(rank)] for i in range(n*m)]
            V_list = [[float(best_V[i, r]) for r in range(rank)] for i in range(m*p)]
            W_list = [[float(best_W[i, r]) for r in range(rank)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition (rank 50)
    rank = 50
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list[i * m + j][column] = 1
                V_list[j * p + k][column] = 1
                W_list[k * n + i][column] = 1
                column += 1
    return {"U": U_list, "V": V_list, "W": W_list}
