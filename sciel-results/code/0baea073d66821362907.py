"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses the schoolbook decomposition with rank = n*m*p = 60.
    The structure is organized by iterating over the output indices (i, k) and input index (j),
    which corresponds to a standard basis for the tensor space.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Pre-allocate lists for U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate through all combinations of i, j, k
    # U corresponds to A (n x m), index i*m + j
    # V corresponds to B (m x p), index j*p + k
    # W corresponds to C (p x n), index k*n + i (trace convention W[k*n+i, r])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
