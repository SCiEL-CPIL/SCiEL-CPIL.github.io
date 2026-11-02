import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    
    rank = m * n  # 3 * 2 = 6
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            c = j * n + i
            U[i * m + j][c] = 1
            for k in range(p):
                V[j * p + k][c] = 1
            for k in range(p):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
