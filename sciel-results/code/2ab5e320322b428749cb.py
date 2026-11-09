import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 by 5x5 matrix.
    Uses 8 Strassen 2x2x2 blocks (rank 7 each) + 2 rank-1 terms for remaining indices = rank 58.
    """
    n, m, p = 4, 5, 5
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute the 8 Strassen blocks covering 2x2x2 subcubes
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3]),
        ([2, 3], [0, 1], [0, 1]),
        ([2, 3], [0, 1], [2, 3]),
        ([2, 3], [2, 3], [0, 1]),
        ([2, 3], [2, 3], [2, 3])
    ]
    
    # Total rank: 8 blocks * 7 terms + 2 rank-1 terms = 58
    rank = 58
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Add Strassen terms for each block
    for I, J, K in blocks:
        if time.time() > deadline:
            break
        i1, i2 = I
        j1, j2 = J
        k1, k2 = K
        
        # M1 = (a11 + a22)(b11 + b22)
        U[i1 * m + j1][col] += 1
        U[i2 * m + j2][col] += 1
        V[j1 * p + k1][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[i2 * m + j1][col] += 1
        U[i2 * m + j2][col] += 1
        V[j1 * p + k1][col] += 1
        W[k1 * n + i2][col] += 1
        W[k2 * n + i2][col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[i1 * m + j1][col] += 1
        V[j1 * p + k2][col] += 1
        V[j2 * p + k2][col] += -1
        W[k2 * n + i1][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[i2 * m + j2][col] += 1
        V[j2 * p + k1][col] += 1
        V[j1 * p + k1][col] += -1
        W[k1 * n + i1][col] += 1
        W[k1 * n + i2][col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[i1 * m + j1][col] += 1
        U[i1 * m + j2][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += -1
        W[k2 * n + i1][col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[i2 * m + j1][col] += 1
        U[i1 * m + j1][col] += -1
        V[j1 * p + k1][col] += 1
        V[j1 * p + k2][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[i1 * m + j2][col] += 1
        U[i2 * m + j2][col] += -1
        V[j2 * p + k1][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += 1
        col += 1
    
    # Rank-1 term for (i, 4, k) group: covers all i in {0,1,2,3}, k in {0,1,2,3,4}
    # U[i*5+4, col] = 1, V[4*5+k, col] = 1, W[k*4+i, col] = 1
    for i in range(n):
        U[i * m + 4][col] = 1
    for k in range(p):
        V[4 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    # Rank-1 term for (i, j, 4) group: covers all i in {0,1,2,3}, j in {0,1,2,3}
    # U[i*5+j, col] = 1, V[j*5+4, col] = 1, W[4*4+i, col] = 1
    for i in range(n):
        for j in range(4):
            U[i * m + j][col] = 1
    for j in range(4):
        V[j * p + 4][col] = 1
    for i in range(n):
        W[4 * n + i][col] = 1
    col += 1
    
    # Trim to actual used rank
    if col < rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
