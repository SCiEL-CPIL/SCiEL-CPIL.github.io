"""Program for rank-minimal tensor decomposition of 2x5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for 2x5x5 matrix multiplication.
    
    Uses the schoolbook decomposition with rank 50. Each rank-1 term corresponds
    to one (i, j, k) triple: A[i,j] * B[j,k] contributes to C[i,k].
    
    Index conventions:
      U: shape (n*m, rank), index i*m + j
      V: shape (m*p, rank), index j*p + k
      W: shape (p*n, rank), index k*n + i
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
