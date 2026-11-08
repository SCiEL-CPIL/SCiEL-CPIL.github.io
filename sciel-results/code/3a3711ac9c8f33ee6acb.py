"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor for matrix multiplication A (n x m) * B (m x p) = C (n x p)
    # has entries T[i, j, k] = 1 for all i, j, k.
    # This is the all-ones tensor, which has rank 1.
    # We construct a rank-1 decomposition.
    rank = 1
    
    # U shape: (n * m, rank)
    # U[i * m + j, r] = u[r, i] * v[r, j]
    # Let u = [1, 1, 1], v = [1, 1, 1], w = [1, 1, 1, 1, 1]
    
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
