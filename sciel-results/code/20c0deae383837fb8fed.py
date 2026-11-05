import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    deadline = time.time() + 140
    
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # Part 2: Three 2x2x2 Strassen blocks
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1
        U[0 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M2
        U[1 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0; W[k1 * n + 1, col] += -1.0
        col += 1
        
        # M3
        U[0 * m + 1, col] += 1.0
        V[1 * p + k1, col] += 1.0; V[2 * p + k1, col] += -1.0
        W[k1 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M4
        U[1 * m + 2, col] += 1.0
        V[2 * p + k0, col] += 1.0; V[1 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0; W[k0 * n + 1, col] += 1.0
        col += 1
        
        # M5
        U[0 * m + 1, col] += 1.0; U[0 * m + 2, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += -1.0; W[k1 * n + 0, col] += 1.0
        col += 1
        
        # M6
        U[1 * m + 1, col] += 1.0; U[0 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0; V[1 * p + k1, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M7
        U[0 * m + 2, col] += 1.0; U[1 * m + 2, col] += -1.0
        V[2 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
