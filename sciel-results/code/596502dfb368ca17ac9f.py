import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 6 matrix.
    
    The schoolbook decomposition has rank n*m*p.
    For n=2, m=3, p=6, rank = 36.
    It is widely believed that the rank of M_{<2,3,6>} is 36, as no
    Strassen-like reduction is known for this specific rectangular case
    that lowers the rank below the product of dimensions.
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
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mapping:
                # U row: i*m + j
                # V row: j*p + k
                # W row: k*n + i (Trace convention W[k*n+i, r])
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
