import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Strategy: Decompose into p independent (n, m, 1) matrix-vector multiplications.
    # For each output column k, we compute C[:, k] = A @ B[:, k].
    # The tensor for (n, m, 1) multiplication has rank n*m.
    # Since the k-th column uses disjoint indices in V and W, the total rank is p * (n*m).
    
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                # U index: i * m + j
                U[i * m + j][r] = 1
                # V index: j * p + k
                V[j * p + k][r] = 1
                # W index: k * n + i (trace convention)
                W[k * n + i][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
