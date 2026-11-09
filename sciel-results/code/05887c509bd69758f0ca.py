DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    R = m
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            if j < R:
                U[i * m + j][j] = 1
                
    for j in range(m):
        for k in range(p):
            if j < R:
                V[j * p + k][j] = 1
                
    for k in range(p):
        for i in range(n):
            for r in range(R):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
