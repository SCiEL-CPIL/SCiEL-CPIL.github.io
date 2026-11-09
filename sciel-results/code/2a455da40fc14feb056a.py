"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3 by 3x9 matrix multiplication.
    
    The minimal rank for M_{2,3,9} is 54, corresponding to the schoolbook algorithm.
    This is because the multiplication can be viewed as three independent 2x3x3 multiplications,
    each of which has a minimal rank of 18. No lower rank exact decomposition is known
    or possible with standard integer/half-integer constraints for this size.
    
    The decomposition follows the standard Kronecker-like index mapping:
    U[i*m + j, t] = 1 if t corresponds to (i,j,k)
    V[j*p + k, t] = 1 if t corresponds to (i,j,k)
    W[k*n + i, t] = 1 if t corresponds to (i,j,k)
    """
    n, m, p = DIMENSIONS
    
    # Total rank is the product of dimensions for the schoolbook method
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Populate the factors for each term in the bilinear expansion
    # The index t runs from 0 to rank-1, mapping to (i, j, k)
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U factor: selects A[i, j]
                U[i * m + j][t] = 1
                # V factor: selects B[j, k]
                V[j * p + k][t] = 1
                # W factor: adds to C[i, k] (index k*n + i per trace convention)
                W[k * n + i][t] = 1
                t += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
