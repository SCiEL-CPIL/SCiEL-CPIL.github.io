import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank is n * m
    rank = n * m
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Indices for rank: r = i * m + j
    for i in range(n):
        for j in range(m):
            r = i * m + j
            
            # U: index (i, j) -> r
            u_idx = i * m + j
            U[u_idx][r] = 1
            
            # V: index (j, k) -> r for all k
            for k in range(p):
                v_idx = j * p + k
                V[v_idx][r] = 1
                
            # W: index (k, i) -> r for all k
            for k in range(p):
                w_idx = k * n + i
                W[w_idx][r] = 1
                
    return {"U": U, "V": V, "W": W}
