"""Matrix multiplication 3x3 x 3x8 rank-minimal exact decomposition."""

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    # Generate indices for the decomposition
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (i, j, k) in enumerate(indices):
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
