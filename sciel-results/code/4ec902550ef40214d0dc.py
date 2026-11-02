import numpy as np

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the matrix multiplication tensor decomposition.
    Shapes:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    n, m, p = 2, 3, 5
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Fill in the schoolbook decomposition
    # Each term in the sum corresponds to a unique (i, j, k) triple
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, col] = 1
                # V index: j*p + k
                V[j * p + k, col] = 1
                # W index: k*n + i (Trace convention W[k*n+i, r])
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
