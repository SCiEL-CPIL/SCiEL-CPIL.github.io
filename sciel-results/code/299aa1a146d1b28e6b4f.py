# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 115

    # Rank 27 Schoolbook Algorithm
    # This is guaranteed to be correct.
    # T[i,j,k] = delta_{j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
        
    for i in range(n):
        for k in range(p):
            for j in range(m):
                u = {(i, j): 1}
                v = {(j, k): 1}
                # W index is k*n + i
                w = {k * n + i: 1}
                add_term(u, v, w)
                
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j][r] = val
        for (j, k), val in v.items():
            V[j * p + k][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
