import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: 120s limit, target < 10s for safety
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition rank n*m*p = 2*3*7 = 42
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Check time
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
