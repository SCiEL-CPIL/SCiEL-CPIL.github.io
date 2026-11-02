import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Uses the standard schoolbook algorithm which is rank n*m*p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                
                # U row index: i*m + j
                U[i * m + j, c] = 1
                
                # V row index: j*p + k
                V[j * p + k, c] = 1
                
                # W row index: k*n + i
                W[k * n + i, c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
