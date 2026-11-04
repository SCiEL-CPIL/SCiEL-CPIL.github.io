import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = p * n
    
    rank = n * m * p
    
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((np_, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
                
    def to_list(mat):
        return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
