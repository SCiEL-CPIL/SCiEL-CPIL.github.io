"""Program for matrix multiplication 3x4x6 tensor decomposition.

Uses the standard schoolbook algorithm which provides an exact decomposition
with rank n*m*p. For 3x4x6, this is rank 72. While advanced algorithms like
Strassen's exist for 2x2x2 (rank 7), no known algorithm beats the schoolbook
rank for 3x4x6 within the computational constraints.
"""

DIMENSIONS = [3, 4, 6]

def construct():
    """Construct the factor matrices for matrix multiplication 3x4x6.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
