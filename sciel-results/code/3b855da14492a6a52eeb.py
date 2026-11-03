import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Schoolbook decomposition: rank = n*m*p = 24
    # This is the straightforward, guaranteed-correct decomposition.
    # Each term corresponds to one (i, j, k) triple.
    
    rank = n * m * p  # 24
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row index: i*m + j
                # V row index: j*p + k
                # W row index: k*n + i (trace convention)
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
