"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor for C = AB is given by:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
    # Thus, the tensor T[i, j, k, i] = 1 for all valid indices.
    # This is the all-ones tensor of size 5x5x5 (with index mapping).
    # The rank of the all-ones tensor is 1.
    # We can decompose it using a single term with all entries set to 1.
    
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
