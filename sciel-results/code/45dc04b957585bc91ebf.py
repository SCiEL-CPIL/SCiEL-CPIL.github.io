import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as numpy arrays for efficiency and cleaner code
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index for U: row i*m + j
                U[i * m + j, col] = 1.0
                # Index for V: row j*p + k
                V[j * p + k, col] = 1.0
                # Index for W: row k*n + i
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
