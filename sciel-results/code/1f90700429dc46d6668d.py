import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The tensor is a direct sum of m identical sub-tensors.
    # Each sub-tensor corresponds to a fixed column j of A and row j of B.
    # The sub-tensor computes C[:, k] += A[:, j] * B[j, k].
    # This is the outer product of a vector of size n and a vector of size p.
    # The rank of the outer product tensor is n * p.
    # Since the input supports for different j are disjoint, the total rank is sum of ranks.
    # Total rank = m * n * p.
    
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (as per trace convention in prompt)
                
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
