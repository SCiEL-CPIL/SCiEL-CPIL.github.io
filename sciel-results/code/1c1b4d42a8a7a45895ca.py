import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition
    # Each multiplication corresponds to a unique (i, j, k) tuple
    # U[i*m + j, t] = 1
    # V[j*p + k, t] = 1
    # W[k*n + i, t] = 1
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][t] = 1
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
                t += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
