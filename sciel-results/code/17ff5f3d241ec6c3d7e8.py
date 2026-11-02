import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # We construct the matrices U, V, W using NumPy for a different algorithmic structure
    # This is still the schoolbook decomposition, but constructed via vectorized operations.
    
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists if required by the interface, or return arrays
    # The seed returned lists of lists. The evaluator likely handles both.
    # To be safe and match the "integer/half-integer" hint, we can return lists.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
