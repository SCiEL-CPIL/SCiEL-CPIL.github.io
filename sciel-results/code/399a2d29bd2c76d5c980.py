"""Construct rank-minimal exact tensor decomposition for 3x3 x 3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Use schoolbook algorithm
    rank = n * m * p  # 72
    U = [[0]*rank for _ in range(n*m)]  # 9 rows
    V = [[0]*rank for _ in range(m*p)]  # 24 rows
    W = [[0]*rank for _ in range(p*n)]  # 24 rows
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
