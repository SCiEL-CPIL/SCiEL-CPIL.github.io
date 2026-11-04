"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    # Initialize lists of lists
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Generate indices for schoolbook decomposition
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for c, (i, j, k) in enumerate(indices):
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        U[i * m + j][c] = 1.0
        V[j * p + k][c] = 1.0
        W[k * n + i][c] = 1.0
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
