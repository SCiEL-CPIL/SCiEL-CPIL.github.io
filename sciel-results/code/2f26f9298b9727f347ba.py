"""Matrix multiplication 3x4x4 tensor decomposition."""
DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    # Naive rank = n*m*p = 48
    # Each term corresponds to one (i,j,k) triple
    rank = n * m * p
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
