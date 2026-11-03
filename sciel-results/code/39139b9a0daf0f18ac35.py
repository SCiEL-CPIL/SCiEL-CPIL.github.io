import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    """
    Returns the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions 3x3 * 3x5.
    
    The decomposition is of rank 45, which is the schoolbook rank.
    While the true minimal rank might be lower (e.g., using Strassen-like 
    decompositions for sub-blocks), the schoolbook decomposition is guaranteed 
    to be valid and exact with integer entries.
    
    U has shape (n*m, rank) = (9, 45)
    V has shape (m*p, rank) = (15, 45)
    W has shape (p*n, rank) = (15, 45)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
