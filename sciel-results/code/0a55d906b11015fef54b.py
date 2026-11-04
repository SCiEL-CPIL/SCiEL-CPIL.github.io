import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We construct the factor matrices U, V, W for the tensor decomposition.
    # The matrix multiplication tensor T has indices (i, j, k) corresponding to
    # A[i, j] * B[j, k] -> C[i, k].
    # A decomposition of rank R corresponds to R slices (i, j, k) combinations.
    # Here we use the full schoolbook decomposition with R = n * m * p.
    
    R = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
    # U has shape (n*m, R)
    # V has shape (m*p, R)
    # W has shape (p*n, R)
    
    # Note: The interface expects integer/half-integer entries.
    # We will store them as floats (0.0 and 1.0) which are valid integers.
    
    U = [[0.0 for _ in range(R)] for _ in range(n * m)]
    V = [[0.0 for _ in range(R)] for _ in range(m * p)]
    W = [[0.0 for _ in range(R)] for _ in range(p * n)]
    
    # Iterate over all combinations of i, j, k
    # Each combination (i, j, k) contributes 1 to:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    # where r is the rank index associated with this (i, j, k) tuple.
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r] = 1.0
                V[v_idx][r] = 1.0
                W[w_idx][r] = 1.0
                
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
