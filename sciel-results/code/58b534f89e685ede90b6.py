import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Standard bilinear decomposition rank
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Map rank index r to (i, j, k)
    # r = i * (m * p) + j * p + k
    for r in range(rank):
        i = r // (m * p)
        j = (r // p) % m
        k = r % p
        
        # U index: i * m + j
        u_idx = i * m + j
        # V index: j * p + k
        v_idx = j * p + k
        # W index: k * n + i
        w_idx = k * n + i
        
        U[u_idx][r] = 1
        V[v_idx][r] = 1
        W[w_idx][r] = 1
        
    return {"U": U, "V": V, "W": W}
