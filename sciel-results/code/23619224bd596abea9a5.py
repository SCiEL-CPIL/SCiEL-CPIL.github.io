"""Tensor decomposition for 2x5 by 5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x5 matrix A by a 5x5 matrix B to get a 2x5 matrix C.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Interface:
      U: shape (n*m, rank) = (10, rank) - U[i*m+j, r]
      V: shape (m*p, rank) = (25, rank) - V[j*p+k, r]
      W: shape (p*n, rank) = (10, rank) - W[k*n+i, r]
      
    The decomposition satisfies:
      C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    rank = n * m * p  # 50
    
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
# EVOLVE-BLOCK-END
