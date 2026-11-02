import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Construct the trivial decomposition
    # The tensor T[i, j, k] = e_{i*m+j} \otimes e_{j*p+k} \otimes e_{k*n+i}
    # We decompose it into rank 1 tensors:
    # U[:, r] \otimes V[:, r] \otimes W[:, r]
    # where for each (i, j, k), r = (i*m + j)*p + k
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = (i * m + j) * p + k
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
