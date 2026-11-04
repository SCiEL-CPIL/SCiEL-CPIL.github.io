import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # We need to map each (i, j, k) to a unique column index c
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    
    # Generate all indices
    # i in [0, n), j in [0, m), k in [0, p)
    # Total count is n*m*p = rank
    
    # Create arrays for i, j, k
    # We can use np.indices or meshgrid
    
    # Efficient way:
    # col index c corresponds to i, j, k where c = i * m * p + j * p + k
    
    cols = np.arange(rank)
    
    # Recover i, j, k from cols
    # c = i * (m*p) + j * p + k
    i = cols // (m * p)
    j = (cols % (m * p)) // p
    k = cols % p
    
    # Calculate row indices
    U_rows = i * m + j
    V_rows = j * p + k
    W_rows = k * n + i
    
    # Set the 1s
    U[U_rows, cols] = 1.0
    V[V_rows, cols] = 1.0
    W[W_rows, cols] = 1.0
    
    # Convert to lists of lists as required by interface
    # The interface expects {'U', 'V', 'W'} with lists.
    # NumPy tolist() is fast.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
