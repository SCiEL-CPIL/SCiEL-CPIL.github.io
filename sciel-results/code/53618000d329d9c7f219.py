import numpy as np

def construct():
    n, m, p = 3, 5, 6
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p

    U = np.zeros((nm, rank_school))
    V = np.zeros((mp, rank_school))
    W = np.zeros((pn, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
