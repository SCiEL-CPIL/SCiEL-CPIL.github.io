import time

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Returns the schoolbook decomposition for matrix multiplication.
    This is the standard exact decomposition with rank n*m*p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    # Indexing:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    # c corresponds to the triple (i, j, k)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
