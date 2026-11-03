import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as numpy arrays of zeros
    # Shapes: (n*m, rank), (m*p, rank), (p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Schoolbook decomposition
    # For each i, j, k, we have a rank-1 contribution
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i, col j of A -> i*m + j
                u_idx = i * m + j
                # V index: row j, col k of B -> j*p + k
                v_idx = j * p + k
                # W index: row k, col i of C? 
                # Convention: W[k*n+i, r]
                # Output C is n x p. Index in flattened C is i*p + k?
                # The task says: "output factor follows the official trace convention W[k*n+i,r]"
                # This implies the W matrix rows are indexed by (k, i) in column-major order for the n x p matrix?
                # Or is it row k, column i?
                # Standard flattening of C (n x p) is i*p + k.
                # But the prompt says W[k*n+i]. This suggests the index is k*n + i.
                # This corresponds to a matrix of size p x n?
                # Let's check the seed code.
                # Seed: W[k * n + i][column] = 1
                # So W row index is k*n + i.
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
