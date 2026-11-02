import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds margin
    
    n, m, p = DIMENSIONS
    rank = n * m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            
            for k_prime in range(p):
                V[j * p + k_prime][col] = 1
            
            for k_double_prime in range(p):
                W[k_double_prime * n + i][col] = 1
            
            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
