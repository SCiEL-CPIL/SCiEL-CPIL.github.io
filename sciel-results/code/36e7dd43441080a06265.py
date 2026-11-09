"""Rewritten program for 3x3 matrix multiplication tensor decomposition."""

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the factor matrices U, V, W for the schoolbook (rank 27)
    decomposition of 3x3 matrix multiplication.
    Note: A rank-23 decomposition is theoretically possible but the
    specific coefficients are not reliably available in this context.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term corresponds to A[i,j] * B[j,k] contributing to C[i,k]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
