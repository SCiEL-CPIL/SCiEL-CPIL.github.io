import time

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Returns the schoolbook decomposition for 4x5 by 5x6 matrix multiplication.
    Rank is n*m*p = 120.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank) -> (20, 120)
    # V shape: (m*p, rank) -> (30, 120)
    # W shape: (p*n, rank) -> (24, 120)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mapping for U: row is i*m + j
                U[i * m + j][column] = 1
                # Index mapping for V: row is j*p + k
                V[j * p + k][column] = 1
                # Index mapping for W: row is k*n + i (Trace convention W[k*n+i, r])
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
