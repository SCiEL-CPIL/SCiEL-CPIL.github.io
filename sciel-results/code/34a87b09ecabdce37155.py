"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-18 tensor decomposition for 2x3 by 3x3 matrix multiplication.
    
    The multiplication is: C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    where A is 2x3, B is 3x3, C is 2x3.
    
    Index conventions:
      U: shape (n*m, rank) = (6, 18), U[i*m + j, r]
      V: shape (m*p, rank) = (9, 18), V[j*p + k, r]
      W: shape (p*n, rank) = (6, 18), W[k*n + i, r]
    
    For each triple (i, j, k) we have one rank-1 term:
      U[i*m+j, r] = 1
      V[j*p+k, r] = 1
      W[k*n+i, r] = 1
    """
    n, m, p = 2, 3, 3
    rank = n * m * p  # 18
    
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
