"""Program for rank-minimal tensor decomposition of 2x5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for 2x5x5 matrix multiplication.
    
    We use the schoolbook decomposition which achieves rank 50.
    The multiplication tensor T_{2,5,5} decomposes as the direct sum of two copies
    of the 1x5x5 tensor. The schoolbook method gives rank n*m*p = 50.
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    rank = n * m * p  # 50
    
    # U: shape (n*m, rank) = (10, 50)
    # V: shape (m*p, rank) = (25, 50)
    # W: shape (p*n, rank) = (10, 50)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook: for each (i, j, k) triple, one rank-1 term
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1  (following the convention W[k*n+i, r])
    
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
