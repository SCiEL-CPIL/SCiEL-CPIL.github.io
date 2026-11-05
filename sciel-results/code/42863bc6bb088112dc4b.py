import numpy as np

def construct():
    n, m, p = 2, 3, 5
    
    # Schoolbook rank is n * m * p
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill in the schoolbook decomposition
    # For each triple (i, j, k), we have a rank-1 term
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    # where c is the column index for this term
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
                
    # Convert to list of lists for the return format
    # The prompt example returns {'U', 'V', 'W'} with lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
