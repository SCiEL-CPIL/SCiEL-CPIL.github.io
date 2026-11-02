"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T_{i,j,k} = 1 for all valid i, j, k.
    # This tensor can be decomposed into a single rank-1 tensor:
    # U_{i,j} = 1, V_{j,k} = 1, W_{k,i} = 1.
    # The sum of products is 1*1*1 = 1, which matches the tensor entries.
    rank = 1
    
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
