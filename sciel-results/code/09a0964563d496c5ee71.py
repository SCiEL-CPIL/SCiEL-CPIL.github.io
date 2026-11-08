import time
import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    U_2x2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [1, 0, 0, -1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, -1, 0]
    ], dtype=float)
    
    V_2x2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, -1],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [1, 0, 0, 0]
    ], dtype=float)
    
    W_2x2 = np.array([
        [1, 0, 0, 1, -1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, -1, 0, 0]
    ], dtype=float)
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for bg in range(2):
        for lp in range(4):
            for s in range(7):
                r = (bg * 4 + lp) * 7 + s
                for i in range(2):
                    for jp in range(2):
                        U[i*m + (lp//2)*2 + jp, r] = U_2x2[s, i*2+jp]
                for jp in range(2):
                    for kp in range(2):
                        V[((lp//2)*2+jp)*p + 2*bg+kp, r] = V_2x2[s, jp*2+kp]
                for i in range(2):
                    for q in range(2):
                        W[(2*lp+q)*n + i, r] = W_2x2[i*2+q, s]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
