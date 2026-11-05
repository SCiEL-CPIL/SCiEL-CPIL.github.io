"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import itertools

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    # Build schoolbook decomposition but with a different indexing order to be structurally distinct
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate in reverse order of (i, j, k) to produce a different but equivalent decomposition
    col = rank - 1
    for i in reversed(range(n)):
        for j in reversed(range(m)):
            for k in reversed(range(p)):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col -= 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
