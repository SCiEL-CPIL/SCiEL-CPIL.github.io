"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The tensor for matrix multiplication A (n x m) * B (m x p) -> C (n x p)
    # is T[i, j, k] = 1 for all i, j, k.
    # We can decompose this into a single rank-1 tensor:
    # T = u v^T w^T where u, v, w are all-ones vectors.
    # This corresponds to U, V, W being all-ones matrices of rank 1.
    
    rank = 1
    
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
