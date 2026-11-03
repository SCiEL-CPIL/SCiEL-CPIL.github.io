import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Trivial decomposition with rank R = n * m * p
    R = n * m * p
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Map r to (i, j, k)
    # r = i * m * p + j * p + k
    # Using numpy for efficient initialization and assignment
    U_np = np.zeros((n * m, R))
    V_np = np.zeros((m * p, R))
    W_np = np.zeros((p * n, R))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_np[i * m + j, r] = 1
                V_np[j * p + k, r] = 1
                W_np[k * n + i, r] = 1
                
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()
                
    return {"U": U, "V": V, "W": W}
