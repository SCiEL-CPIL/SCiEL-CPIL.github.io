"""Matrix multiplication 3,3,4 tensor decomposition."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook exact tensor decomposition for 3x3 times 3x4 matrix multiplication.
    Rank = n*m*p = 36, with integer entries.
    
    The multiplication tensor T_{i,j,k} = sum_{a,b} A_{i,a} * B_{a,b} * (output at position k*n+i for the W convention)
    Actually, let me re-read the convention: W[k*n+i, r]
    
    For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p.
    C_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    The tensor is T[i, j, k] where i indexes A's rows, j indexes the shared dimension, k indexes B's columns.
    The output is C[i, k].
    
    The factor matrices are:
    - U: shape (n*m, rank) - indexes (i, j) pairs from A
    - V: shape (m*p, rank) - indexes (j, k) pairs from B  
    - W: shape (p*n, rank) - indexes (k, i) pairs for output C[i,k] but stored as W[k*n+i]
    
    So for each rank-1 term r, we have:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = C[i,k]
    
    For schoolbook: each (i,j,k) triple gives one rank-1 term where:
    U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    and all other entries in that column are 0.
    """
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
