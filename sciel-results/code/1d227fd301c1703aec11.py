import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14

    rank_sb = n * m * p  # 42

    U_sb = np.zeros((nm, rank_sb), dtype=np.int64)
    V_sb = np.zeros((mp, rank_sb), dtype=np.int64)
    W_sb = np.zeros((pn, rank_sb), dtype=np.int64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * (m * p) + j * p + k
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1

    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
