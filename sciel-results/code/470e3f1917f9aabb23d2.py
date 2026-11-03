"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Construct a valid exact tensor decomposition for 2x5x6 matrix multiplication.
    # We use the standard schoolbook (outer product) decomposition of rank n*m*p = 60.
    # The interface requires:
    #   U: (n*m, rank)  ->  (10, rank)
    #   V: (m*p, rank)  ->  (30, rank)
    #   W: (p*n, rank)  ->  (12, rank)
    # with the trace convention W[k*n + i, r].
    #
    # The decomposition identity is:
    #   sum_{i,j,k} A_{i,j} * B_{j,k} * (E_{i,k} in C)
    #   = sum_{c=0}^{rank-1}  (U_c outer V_c) outer W_c  (appropriately transposed)
    #
    # For the standard decomposition, each term c corresponds to a triple (i,j,k):
    #   U[i*m + j, c] = 1
    #   V[j*p + k, c] = 1
    #   W[k*n + i, c] = 1
    # All other entries are 0.
    #
    # This yields the exact matrix multiplication tensor and is always valid.
    n, m, p = DIMENSIONS
    rank = n * m * p
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
