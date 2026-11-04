"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-24 (schoolbook) decomposition for 2x3 x 3x4 matrix multiplication.
    
    The schoolbook decomposition uses one rank-1 term per (i,j,k) triple,
    giving rank = n*m*p = 24.
    
    U shape: (n*m, rank) = (6, 24)
    V shape: (m*p, rank) = (12, 24)
    W shape: (p*n, rank) = (8, 24)
    
    For column (i,j,k):
      U[i*m+j, col] = 1
      V[j*p+k, col] = 1
      W[k*n+i, col] = 1
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    rank = n * m * p  # 24
    
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
