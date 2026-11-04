import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank: 3*3*8 = 72
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
    
    # Verify
    error = 0
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                s_val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
                i_u = u // m
                j_u = u % m
                j_v = v // p
                k_v = v % p
                k_w = w // n
                i_w = w % n
                if i_u == i_w and k_v == k_w and j_u == j_v:
                    target = 1
                else:
                    target = 0
                error += (s_val - target) ** 2
    
    if error < 1e-9:
        return {"U": U, "V": V, "W": W}
    
    raise RuntimeError("Schoolbook verification failed")
