"""
Program to find the rank-minimal exact tensor decomposition for
multiplying a 2 by 3 matrix by a 3 by 4 matrix.
"""

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the matrix multiplication tensor.
    The rank is n*m*p = 24.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i, j, c] = 1
                U[i * m + j][col] = 1
                # V[j, k, c] = 1
                V[j * p + k][col] = 1
                # W[k, i, c] = 1 (Note: W index is k*n + i)
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
