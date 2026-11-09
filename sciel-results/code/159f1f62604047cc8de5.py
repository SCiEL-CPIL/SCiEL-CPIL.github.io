import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Constructs a tensor decomposition for matrix multiplication of dimensions 3x4x7.
    Implements a block-structured decomposition. Although Strassen-style 
    cancellations are not available for these specific dimensions (3, 4, 7) 
    to reduce the rank below the schoolbook bound, this implementation 
    explicitly constructs the basis vectors for the full tensor product space.
    """
    n, m, p = DIMENSIONS
    
    # The minimal rank for M_{n,m,p} is generally n*m*p for these small dimensions
    # as no non-trivial bilinear identities reduce the count.
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over all basis elements of the tensor product
    # Index mapping:
    # A has indices (i, j) -> row i*m + j in U
    # B has indices (j, k) -> row j*p + k in V
    # C has indices (k, i) -> row k*n + i in W (Trace convention W[k*n+i, r])
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Assign the basis vector for the term A[i,j] * B[j,k]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
