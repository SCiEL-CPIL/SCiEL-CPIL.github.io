"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-125 exact tensor decomposition for 5x5 x 5x5 matrix multiplication.
    This is the schoolbook decomposition, which is known to be optimal for
    general 5x5 matrix multiplication under standard conventions.
    
    The decomposition satisfies:
    C[k*n + i, j] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    where the output follows the trace convention W[k*n+i, r].
    
    In this decomposition, each term (i, j, k) contributes exactly one
    rank-1 term with U=1, V=1, W=1 for the corresponding indices.
    """
    n, m, p = DIMENSIONS  # n=5, m=5, p=5
    rank = n * m * p      # 125
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]  # shape (25, 125)
    V = [[0] * rank for _ in range(m * p)]  # shape (25, 125)
    W = [[0] * rank for _ in range(p * n)]  # shape (25, 125)
    
    # Fill in the rank-1 terms
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
