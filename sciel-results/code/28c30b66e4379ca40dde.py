"""
Rewritten program using NumPy for matrix construction.
Implements the schoolbook decomposition for 3x3x4 matrix multiplication.
The rank is 36.
"""
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the factor matrices U, V, W for the matrix multiplication tensor.
    n=3, m=3, p=4.
    U shape: (n*m, rank) = (9, 36)
    V shape: (m*p, rank) = (12, 36)
    W shape: (p*n, rank) = (12, 36)
    
    The decomposition satisfies:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is a valid 
    multiplication term, which is always 1 in the schoolbook sense where each 
    (i,j,k) triple is its own term.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # Iterate over all possible (i, j, k) indices for the schoolbook decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i, col j -> i*m + j
                u_idx = i * m + j
                # Index in V: row j, col k -> j*p + k
                v_idx = j * p + k
                # Index in W: row k, col i -> k*n + i (as per task description W[k*n+i,r])
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
