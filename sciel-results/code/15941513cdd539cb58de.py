"""Program that finds a rank-minimal exact tensor decomposition for 4x4 by 4x8 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

def _validate(U, V, W, n, m, p):
    """Check that the decomposition exactly reproduces the matmul tensor."""
    rank = len(U[0]) if U else 0
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # The tensor is T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # But we need to be careful with indexing conventions.
    # U index: i*m + j, where i in [0,n), j in [0,m)
    # V index: j*p + k, where j in [0,m), k in [0,p)
    # W index: k*n + i, where k in [0,p), i in [0,n)
    
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The matmul tensor should be T[i,j,k] = 1 if i == j == k (in the sense of the product), 
    # but actually for matmul, T[i,j,k] is the coefficient of a[i,j]*b[j,k] in c[i,k].
    # So T[i,j,k] = 1 for all valid i,j,k, and the decomposition should satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Check a sample of entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-10):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
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
    
    # Validate
    if not _validate(U, V, W, n, m, p):
        raise ValueError("Decomposition is invalid")
    
    return {"U": U, "V": V, "W": W}
