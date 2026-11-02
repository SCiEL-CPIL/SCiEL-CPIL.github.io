import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_sb[u_idx, col] = 1
                V_sb[v_idx, col] = 1
                W_sb[w_idx, col] = 1
                col += 1
    
    # Try to use 23-rank for 3x3x3 part
    # 3x3x2 part uses schoolbook (18)
    # Total 23 + 18 = 41
    # I need the 23 terms.
    # Since I don't have them, I return schoolbook.
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
