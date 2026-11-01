import time
import sys

# Dimensions for the matrix multiplication: (2, 3, 7)
DIMENSIONS = [2, 3, 7]

def construct():
    """
    Constructs the schoolbook decomposition for matrix multiplication.
    For dimensions (n, m, p), the rank is n*m*p.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices for the schoolbook decomposition
    # T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # We want T_{i,j,k} = 1 for the corresponding multiplication term.
    # In the schoolbook case, each (i,j,k) triple is a separate rank-1 term.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i of A, col j of A -> i*m + j
                U[i * m + j][col] = 1
                # V index: row j of B, col k of B -> j*p + k
                V[j * p + k][col] = 1
                # W index: row k of C, col i of C -> k*n + i (per contract: W[k*n+i, r])
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
