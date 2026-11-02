"""Tensor decomposition for 3x4x5 matrix multiplication."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-60 schoolbook tensor decomposition for 3x4x5 matrix multiplication.
    
    The multiplication tensor T has entries T[i, j, k] = 1 where
    C[i,k] = sum_j A[i,j] * B[j,k].
    
    We factor T as sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] where:
      U has shape (n*m, rank) = (12, 60)
      V has shape (m*p, rank) = (20, 60)  
      W has shape (p*n, rank) = (15, 60)
    
    Index conventions:
      U[i*m + j, c]
      V[j*p + k, c]
      W[k*n + i, c]
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    rank = n * m * p       # 60
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
