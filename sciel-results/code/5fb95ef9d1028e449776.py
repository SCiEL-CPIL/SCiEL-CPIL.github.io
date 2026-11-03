DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T_{i,j,k} = 1 for all i,j,k
    # because C_{ik} = sum_j A_{ij} B_{jk}.
    # Each term A_{ij} B_{jk} has coefficient 1 in C_{ik}.
    # Thus T is the all-ones tensor.
    # The all-ones tensor has rank 1.
    # Decomposition: U_{i,j} = 1, V_{j,k} = 1, W_{k,i} = 1.
    
    rank = 1
    
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
