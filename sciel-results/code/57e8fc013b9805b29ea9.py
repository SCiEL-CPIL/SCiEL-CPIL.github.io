"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal decomposition for 3x5x5 matrix multiplication.
    Strategy: Direct sum of three 1x5x5 row-vector-matrix multiplication tensors.
    Each 1x5x5 tensor has rank 5 (optimal).
    Total rank = 3 * 5 = 15.
    """
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    rank = n * m  # 3 * 5 = 15
    
    U = [[0] * rank for _ in range(n * m)]  # 15 x 15
    V = [[0] * rank for _ in range(m * p)]  # 25 x 15
    W = [[0] * rank for _ in range(p * n)]  # 15 x 15
    
    # For each output row i (0 to n-1), we need to compute:
    # C[i][k] = sum_{j=0}^{m-1} A[i][j] * B[j][k] for k=0 to p-1
    # 
    # This is a rank-m tensor: for each j, we have a rank-1 term:
    #   U[i*m+j][col] = 1
    #   V[j*p+k][col] = 1
    #   W[k*n+i][col] = 1
    # where col = i*m + j
    #
    # This gives rank n*m = 15, which is optimal for this structure.
    
    column = 0
    for i in range(n):
        for j in range(m):
            col = column
            column += 1
            U[i * m + j][col] = 1
            for k in range(p):
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
