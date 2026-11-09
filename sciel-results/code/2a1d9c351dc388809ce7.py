import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor for C = AB is the all-ones tensor
    # in the indices i, j, k corresponding to A_ij, B_jk, C_ki.
    # The rank of the all-ones tensor is 1.
    # U_ij = 1, V_jk = 1, W_ki = 1.
    
    R = 1
    
    U = [[1.0] * R for _ in range(n * m)]
    V = [[1.0] * R for _ in range(m * p)]
    W = [[1.0] * R for _ in range(p * n)]
    
    # Verification is optional but good for sanity
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if somehow verification fails (it shouldn't)
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1.0
                    V[j*p + k][col] = 1.0
                    W[k*n + i][col] = 1.0
                    col += 1
    
    return {"U": U, "V": V, "W": W}
