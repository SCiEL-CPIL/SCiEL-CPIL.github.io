"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 12 decomposition
    rank = 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for j in range(m):
        # Pair k=0,1 and k=2,3
        # Pair 1: (0, j, 0) and (1, j, 1)
        # Pair 2: (0, j, 1) and (1, j, 0)
        # Pair 3: (0, j, 2) and (1, j, 3)
        # Pair 4: (0, j, 3) and (1, j, 2)
        
        pairs = [
            ((0, j, 0), (1, j, 1)),
            ((0, j, 1), (1, j, 0)),
            ((0, j, 2), (1, j, 3)),
            ((0, j, 3), (1, j, 2))
        ]
        
        for (i1, j1, k1), (i2, j2, k2) in pairs:
            # U
            U[i1 * m + j1][r] = 1
            U[i2 * m + j2][r] = 1
            # V
            V[j1 * p + k1][r] = 1
            V[j2 * p + k2][r] = 1
            # W
            W[k1 * n + i1][r] = 1
            W[k2 * n + i2][r] = 1
            # Cross terms W must be 0 (already 0)
            # W[k2 * n + i1][r] = 0
            # W[k1 * n + i2][r] = 0
            r += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
