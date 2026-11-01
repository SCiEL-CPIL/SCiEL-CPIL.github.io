"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The decomposition is exact for the schoolbook algorithm.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n * m * p
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n * m, rank)
    # V shape: (m * p, rank)
    # W shape: (p * n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    # The matrix multiplication tensor T_{i, j, k} is 1 if we are computing
    # C_{ik} using A_{ij} and B_{jk}.
    # In the decomposition C = U * diag(1) * V^T * W^T (conceptually),
    # each rank-1 term corresponds to a single (i, j, k) interaction.
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                U[i * m + j][column] = 1
                # V index: j * p + k
                V[j * p + k][column] = 1
                # W index: k * n + i (as per trace convention W[k*n+i,r])
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
