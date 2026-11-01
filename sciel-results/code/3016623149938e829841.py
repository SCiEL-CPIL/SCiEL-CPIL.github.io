import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor for multiplying an n x m matrix by an m x p matrix
    # decomposes into p independent tensors for multiplying n x m by m x 1.
    # The rank of M_{n,m,1} is n*m.
    # Thus the total rank is p * (n * m).
    # For n=2, m=2, p=12, rank = 12 * 4 = 48.
    
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Construct the decomposition
    # We iterate over each column k of the second matrix (and corresponding row in W)
    # and each (i, j) pair in the first matrix.
    # This effectively creates the schoolbook decomposition.
    
    col_idx = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                # U index: i * m + j
                u_idx = i * m + j
                # V index: j * p + k
                v_idx = j * p + k
                # W index: k * n + i (official trace convention)
                w_idx = k * n + i
                
                U[u_idx, col_idx] = 1
                V[v_idx, col_idx] = 1
                W[w_idx, col_idx] = 1
                col_idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
