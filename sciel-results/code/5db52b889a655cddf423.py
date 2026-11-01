import numpy as np
import random

def construct():
    n, m, p = 2, 3, 4
    num_eq = n * m * p
    # Variables for W are indexed by (k, i, r)
    # Total W vars = p * n * R
    # Equations indexed by (i, j, k)
    
    # Try ranks from 20 to 23
    # If none found, return schoolbook 24
    for R in range(20, 24):
        # Try a few random initializations
        for _ in range(100):
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            
            # Build A matrix
            # Rows: n*m*p = 24
            # Cols: p*n*R
            A = np.zeros((num_eq, p * n * R))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * (m * p) + j * p + k
                        for r in range(R):
                            # W index is (k, i, r)
                            # Flat index: (k * n + i) * R + r
                            w_idx = (k * n + i) * R + r
                            A[eq_idx, w_idx] = U[i * m + j, r] * V[j * p + k, r]
            
            b = np.ones(num_eq)
            
            # Solve A W = b
            # Use least squares
            try:
                W_sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            # Check if W_sol is close to integer
            W_int = np.round(W_sol).astype(int)
            
            # Verify exactness
            # Recalculate A @ W_int
            res = A @ W_int
            if np.allclose(res, b, atol=1e-9):
                # Found a valid decomposition
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_int.tolist()
                }
                
    # Fallback to Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
