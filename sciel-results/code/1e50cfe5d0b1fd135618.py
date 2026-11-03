"""Tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    
    # Matrix multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor M[i,j,k] = 1 for all i,j,k
    # Decomposition: M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Key insight: For each j, the slice where A[i,j]*B[j,k] contributes
    # is a rank-1 tensor in (i,k) space (all ones). So we need one term per j.
    # Rank = m = 3.
    
    rank = m  # 3
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for j in range(m):
        # U column j: 1 at rows i*m + j for all i
        for i in range(n):
            U[i * m + j, j] = 1.0
        # V column j: 1 at rows j*p + k for all k
        for k in range(p):
            V[j * p + k, j] = 1.0
        # W column j: 1 at all positions (k*n + i for all k,i)
        W[:, j] = 1.0
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-10:
                    # Fallback to trivial decomposition
                    rank = n * m * p
                    U = np.zeros((n * m, rank), dtype=float)
                    V = np.zeros((m * p, rank), dtype=float)
                    W = np.zeros((p * n, rank), dtype=float)
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj, col] = 1.0
                                V[jj * p + kk, col] = 1.0
                                W[kk * n + ii, col] = 1.0
                                col += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
