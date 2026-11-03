"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-54 exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    This is the schoolbook decomposition.
    Rank = n * m * p = 2 * 3 * 9 = 54.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists (or numpy arrays if preferred, but lists are fine for small sizes)
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
