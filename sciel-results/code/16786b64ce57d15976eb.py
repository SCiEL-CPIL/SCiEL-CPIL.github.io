"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank 50 decomposition for 2x5x5 matrix multiplication.
    # Uses schoolbook (rank 50) for all terms.
    n, m, p = DIMENSIONS
    
    # We will build the U, V, W matrices.
    # U: (n*m, rank) = (10, rank)
    # V: (m*p, rank) = (25, rank)
    # W: (p*n, rank) = (10, rank)
    
    # Start with empty lists of columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    # Schoolbook decomposition: for each (i, j, k), add a rank-1 term
    # A[i,j] * B[j,k] contributes to C[i,k]
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i (trace convention)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                add_term(u, v, w)
            
    # Convert lists to matrices
    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
