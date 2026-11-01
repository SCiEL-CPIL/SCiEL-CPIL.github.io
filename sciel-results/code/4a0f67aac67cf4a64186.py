"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 2x4 * 4x7 -> 2x7
    Using schoolbook decomposition with rank = n*m*p = 56.
    
    Convention:
      U has shape (n*m, rank) = (8, rank), indexed as U[i*m + j, t]
      V has shape (m*p, rank) = (28, rank), indexed as V[j*p + k, t]
      W has shape (p*n, rank) = (14, rank), indexed as W[k*n + i, t]
    
    For each (i,j,k), we have a rank-1 term:
      U[i*m+j, t] = 1
      V[j*p+k, t] = 1
      W[k*n+i, t] = 1
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][t] = 1
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
                t += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
