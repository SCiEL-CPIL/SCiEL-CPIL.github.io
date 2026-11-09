import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 120

def construct():
    n, m, p = 4, 4, 9
    # Split B into B1 (4x4), B2 (4x4), B3 (4x1)
    # Use schoolbook for 4x4x4 (rank 64) and 4x4x1 (rank 4)
    # Total rank 64 + 64 + 4 = 132
    
    rank1 = 64
    rank2 = 64
    rank3 = 4
    total_rank = rank1 + rank2 + rank3
    
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # B1: columns 0-3
    # B2: columns 4-7
    # B3: column 8
    
    col = 0
    # B1
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    # B2
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    # B3
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1
            V[j*p+8, col] = 1
            W[8*n+i, col] = 1
            col += 1
            
    return {'U': U, 'V': V, 'W': W}
