import numpy as np

def construct():
    n, m, p = 4, 5, 6
    
    # Total rank: 64 (4x4x4) + 32 (4x4x2) + 1 (4x1x6) = 97
    rank = 97
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A1 (4x4) @ B1a (4x4) -> contributes to C[i,k] for i=0..3, k=0..3
    # Schoolbook: 4*4*4 = 64 terms
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Part 2: A1 (4x4) @ B1b (4x2) -> contributes to C[i,k] for i=0..3, k=4..5
    # Schoolbook: 4*4*2 = 32 terms
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Part 3: A2 (4x1) @ B2 (1x6) -> contributes to C[i,k] for i=0..3, k=0..5
    # Rank 1: outer product
    for i in range(4):
        U[i * m + 4, col] = 1  # A[i,4]
    for k in range(6):
        V[4 * p + k, col] = 1  # B[4,k]
    for k in range(6):
        for i in range(4):
            W[k * n + i, col] = 1
    col += 1
    
    # Convert to lists as required
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
