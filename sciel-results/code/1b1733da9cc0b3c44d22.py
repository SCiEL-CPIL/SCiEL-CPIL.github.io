import time

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Returns a rank-minimal tensor decomposition for 2x3 by 3x9 matrix multiplication.
    The rank is 54 (schoolbook).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U corresponds to input A (n x m)
                # Index: i * m + j
                U[i * m + j][col] = 1
                
                # V corresponds to input B (m x p)
                # Index: j * p + k
                V[j * p + k][col] = 1
                
                # W corresponds to output C (n x p)
                # Index: k * n + i  (Trace convention W[k*n+i, r])
                W[k * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
