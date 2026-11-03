import time

DIMENSIONS = [3, 3, 5]

def construct():
    """
    Returns the schoolbook tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    
    The multiplication tensor T[i,j,k] (i=0..2, j=0..2, k=0..4) is decomposed into rank-1 terms.
    The schoolbook method uses one rank-1 term for each (i, j, k) triple.
    Total rank = 3 * 3 * 5 = 45.
    
    U has shape (9, 45), V has shape (15, 45), W has shape (15, 45).
    Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    rank = n * m * p  # 45
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
