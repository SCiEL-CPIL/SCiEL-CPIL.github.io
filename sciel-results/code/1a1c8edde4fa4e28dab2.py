import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 49
    
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    u_inner = np.zeros((7, 4))
    v_inner = np.zeros((7, 4))
    w_inner = np.zeros((7, 4))
    
    u_inner[0] = [1, 0, 0, 1]
    v_inner[0] = [1, 0, 0, 1]
    u_inner[1] = [0, 0, 1, 1]
    v_inner[1] = [1, 0, 0, 0]
    u_inner[2] = [1, 0, 0, 0]
    v_inner[2] = [0, 1, 0, -1]
    u_inner[3] = [0, 0, 0, 1]
    v_inner[3] = [0, 0, 1, -1]
    u_inner[4] = [1, 1, 0, 0]
    v_inner[4] = [0, 0, 0, 1]
    u_inner[5] = [-1, 0, 1, 0]
    v_inner[5] = [1, 1, 0, 0]
    u_inner[6] = [0, 1, 0, -1]
    v_inner[6] = [0, 0, 1, 1]
    
    beta = np.zeros((7, 4))
    beta[0, 0] = 1
    beta[0, 3] = 1
    beta[1, 2] = 1
    beta[1, 3] = -1
    beta[2, 1] = 1
    beta[2, 3] = 1
    beta[3, 0] = 1
    beta[3, 2] = 1
    beta[4, 0] = -1
    beta[4, 1] = 1
    beta[5, 3] = 1
    beta[6, 0] = 1
    
    alpha = np.zeros((7, 2, 2, 2, 2))
    gamma = np.zeros((7, 2, 2, 2, 2))
    
    alpha[0, 0, 0, 0, 0] = 1
    alpha[0, 0, 0, 1, 1] = 1
    gamma[0, 0, 0, 0, 0] = 1
    gamma[0, 0, 0, 1, 1] = 1
    
    alpha[1, 0, 0, 1, 0] = 1
    alpha[1, 0, 0, 1, 1] = 1
    gamma[1, 0, 0, 0, 0] = 1
    
    alpha[2, 0, 0, 0, 0] = 1
    gamma[2, 0, 0, 0, 1] = 1
    gamma[2, 0, 0, 1, 1] = -1
    
    alpha[3, 0, 0, 1, 1] = 1
    gamma[3, 0, 0, 1, 0] = 1
    gamma[3, 0, 0, 0, 0] = -1
    
    alpha[4, 0, 0, 0, 0] = 1
    alpha[4, 0, 0, 0, 1] = 1
    gamma[4, 0, 0, 1, 1] = 1
    
    alpha[5, 0, 0, 1, 0] = 1
    alpha[5, 0, 0, 0, 0] = -1
    gamma[5, 0, 0, 0, 0] = 1
    gamma[5, 0, 0, 0, 1] = 1
    
    alpha[6, 0, 0, 0, 1] = 1
    alpha[6, 0, 0, 1, 1] = -1
    gamma[6, 0, 0, 1, 0] = 1
    gamma[6, 0, 0, 1, 1] = 1
    
    for k in range(7):
        for l in range(7):
            col = k * 7 + l
            
            for a in range(4):
                if u_inner[l, a] != 0:
                    p = a // 2
                    q = a % 2
                    for i in range(2):
                        for j in range(2):
                            coeff = u_inner[l, a] * alpha[k, p, q, i, j]
                            if coeff != 0:
                                for r in range(2):
                                    for s in range(2):
                                        row_idx = (i * 2 + r) * 4 + (j * 2 + s)
                                        U[row_idx, col] += coeff
                                        
            for b in range(4):
                if v_inner[l, b] != 0:
                    p = b // 2
                    q = b % 2
                    for i in range(2):
                        for j in range(2):
                            coeff = v_inner[l, b] * gamma[k, p, q, i, j]
                            if coeff != 0:
                                for r in range(2):
                                    for s in range(2):
                                        row_idx = (i * 2 + r) * 4 + (j * 2 + s)
                                        V[row_idx, col] += coeff
                                        
            for c in range(4):
                if w_inner[l, c] != 0:
                    p = c // 2
                    q = c % 2
                    for i in range(2):
                        for j in range(2):
                            coeff = w_inner[l, c] * beta[k, i * 2 + j]
                            if coeff != 0:
                                row_idx = (i * 2 + p) * 4 + (j * 2 + q)
                                W[row_idx, col] += coeff

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
