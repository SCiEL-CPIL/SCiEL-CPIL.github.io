"""Matrix multiplication 2,3,9 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor decomposes into two independent 1 x 3 x 9 multiplication tensors.
    # The rank of a 1 x m x p multiplication tensor is m * p.
    # Here, rank for one block is 3 * 9 = 27.
    # Total rank is 2 * 27 = 54.
    
    rank_per_block = m * p
    total_rank = n * rank_per_block
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    for i in range(n):
        # For each row i of A (which is 1 x m), we process the 1 x m x p multiplication
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
