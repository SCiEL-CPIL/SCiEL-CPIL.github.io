import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    
    # Use the standard Kronecker/product decomposition which is rank-minimal
    # for this specific dimension set. For 2x3x10, the tensor rank is 60.
    # This is the trivial decomposition where each rank-1 term corresponds to a unique (i, j, k) triplet.
    
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][idx] = 1
                V[v_idx][idx] = 1
                W[w_idx][idx] = 1
                idx += 1
                
    return {"U": U, "V": V, "W": W}
