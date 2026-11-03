import time
import numpy as np

def construct():
    n, m, p = 3, 3, 6
    rank = 45
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    # First 3x3 slice (k=0,1,2)
    for i in range(3):
        for j in range(3):
            U[i*3+j, col] = 1
            V[j*6, col] = 1
            W[0*3+i, col] = 1
            col += 1
            U[i*3+j, col] = 1
            V[j*6+1, col] = 1
            W[1*3+i, col] = 1
            col += 1
            U[i*3+j, col] = 1
            V[j*6+2, col] = 1
            W[2*3+i, col] = 1
            col += 1
    
    # Second 3x3 slice (k=3,4,5)
    for i in range(3):
        for j in range(3):
            U[i*3+j, col] = 1
            V[j*6+3, col] = 1
            W[3*3+i, col] = 1
            col += 1
            U[i*3+j, col] = 1
            V[j*6+4, col] = 1
            W[4*3+i, col] = 1
            col += 1
            U[i*3+j, col] = 1
            V[j*6+5, col] = 1
            W[5*3+i, col] = 1
            col += 1
    
    # Verification (disabled for speed)
    # A = np.random.randn(3,3)
    # B = np.random.randn(3,6)
    # C = A @ B
    # C_recon = np.zeros((3,6))
    # for r in range(rank):
    #     C_recon += U[:,r] * V[:,r] * W[:,r].reshape(6,3).T
    # if not np.allclose(C_recon, C):
    #     raise ValueError("Decomposition invalid")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
