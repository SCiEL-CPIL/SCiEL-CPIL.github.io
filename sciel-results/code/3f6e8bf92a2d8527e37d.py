"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    
    The tensor T[i, j, k] = 1 for all (i, j, k) in [0,3) x [0,5) x [0,5) represents
    the matrix multiplication C[i,k] = sum_j A[i,j] * B[j,k].
    
    The schoolbook decomposition has rank = 3*5*5 = 75, where each term corresponds
    to a single (i,j,k) triple.
    
    We use the schoolbook decomposition as it is exact and achieves rank 75.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # shape (n*m, rank)
    V = [[0] * rank for _ in range(m * p)]  # shape (m*p, rank)
    W = [[0] * rank for _ in range(p * n)]  # shape (p*n, rank)
    
    # Fill in the rank-1 terms following the trace convention W[k*n+i, r]
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
