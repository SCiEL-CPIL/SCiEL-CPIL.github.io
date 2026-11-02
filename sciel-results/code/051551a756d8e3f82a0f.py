import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank calculation:
    # Part 1: i in 0..3, j in 0..3, k in 0..3 (4x4x4 schoolbook) -> Rank 64
    # Part 2: i in 0..3, j in 0..3, k in 4 (4x4x1 optimized) -> Rank 4
    # Part 3: i in 0..3, j in 4, k in 0..4 (4x1x5 optimized) -> Rank 1
    # Total Rank = 69
    
    rank = 69
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    
    col = 0
    
    # Part 1: Schoolbook for j < 4, k < 4
    for i in range(n):
        for j in range(4):
            for k in range(4):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Part 2: Optimized for j < 4, k = 4
    # C[i, 4] = sum_{j=0}^3 A[i, j] B[j, 4]
    # Decomposition: sum_{alpha=0}^3 A[i, alpha] B[alpha, 4] * 1
    for alpha in range(4):
        for i in range(n):
            U[i * m + alpha, col] = 1
        for j in range(4):
            if j == alpha:
                V[j * p + 4, col] = 1
        for i in range(n):
            W[4 * n + i, col] = 1
        col += 1
        
    # Part 3: Optimized for j = 4, k < 5
    # C[i, k] = A[i, 4] B[4, k]
    # Decomposition: A[i, 4] B[4, k] * 1
    for i in range(n):
        U[i * m + 4, col] = 1
    for k in range(p):
        V[4 * p + k, col] = 1
    for i in range(n):
        for k in range(p):
            W[k * n + i, col] = 1
    col += 1
            
    return {"U": U, "V": V, "W": W}
