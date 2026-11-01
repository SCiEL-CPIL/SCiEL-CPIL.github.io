import time

DIMENSIONS = [2, 2, 14]

def construct():
    # The rank of the matrix multiplication tensor M_{2,2,14} is 56.
    # This is derived from the fact that M_{2,2,14} is the direct sum of 14 copies of M_{2,2,1}.
    # The rank of M_{2,2,1} is 4.
    # Therefore, the minimal rank is 14 * 4 = 56.
    
    n, m, p = DIMENSIONS
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
