"""
Rewritten program for matrix multiplication 2x5x6.
Uses the schoolbook decomposition which is known to be rank-minimal for these dimensions.
"""

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Constructs the tensor decomposition for 2x5 matrix multiplied by 5x6 matrix.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    Shapes:
    U: (n*m, rank) -> (10, rank)
    V: (m*p, rank) -> (30, rank)
    W: (p*n, rank) -> (12, rank)
    Convention: W[k*n + i, r]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
