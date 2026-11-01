"""Matrix multiplication 2,3,9 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The tensor decomposes into a direct sum of n independent 1xm x mxp multiplications.
    # For 1xm x mxp, the rank is m*p.
    # Total rank is n * m * p.
    rank_per_block = m * p
    total_rank = n * rank_per_block
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Construct block-diagonal structures
    # For each row i of the input matrix A (dimension n):
    for i in range(n):
        # Offset in the rank dimension for this block
        block_offset = i * rank_per_block
        
        # Iterate over the m*p terms in this block
        for j in range(m):
            for k in range(p):
                col = block_offset + j * p + k
                
                # U index: i*m + j
                U[i * m + j, col] = 1
                # V index: j*p + k
                V[j * p + k, col] = 1
                # W index: k*n + i
                W[k * n + i, col] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
