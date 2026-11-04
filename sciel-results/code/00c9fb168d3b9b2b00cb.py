import time
import numpy as np

def construct():
    n, m, p = 4, 4, 7
    # Trivial decomposition: rank = n * m * p = 112
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
                
    return {"U": U, "V": V, "W": W}
