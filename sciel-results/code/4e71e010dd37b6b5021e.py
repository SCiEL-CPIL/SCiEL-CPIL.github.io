"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert (n, m, p) == (3, 3, 3)
    
    # Schoolbook decomposition for 3x3 matrix multiplication (rank 27)
    # This is a valid but non-minimal decomposition.
    # U has shape (9, 27), V has shape (9, 27), W has shape (9, 27)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    rank = n * m * p  # 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
