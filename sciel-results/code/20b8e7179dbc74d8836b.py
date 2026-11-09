import time
import numpy as np

def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 110.0
    
    # 8 blocks * 7 Strassen terms + 3 cross terms = 59
    rank_estimate = 60
    U = [[0] * rank_estimate for _ in range(n * m)]
    V = [[0] * rank_estimate for _ in range(m * p)]
    W = [[0] * rank_estimate for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 4x4x4 core (i=0..3, j=0..3, k=0..3)
    # Split into 2x2 blocks
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
    
    for I, J, K in blocks:
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
    
    # Cross term: j=4 (all k)
    # T[i,4,k] = 1 for all i=0..3, k=0..4
    # Rank-1: U[i*m+4]=1, V[4*p+k]=1, W[k*n+i]=1
    for i in range(4):
        U[i * m + 4][col] = 1
    for k in range(5):
        V[4 * p + k][col] = 1
    for k in range(5):
        for i in range(4):
            W[k * n + i][col] = 1
    col += 1
    
    # Cross term: k=4 (all j)
    # T[i,j,4] = 1 for all i=0..3, j=0..4
    # Rank-1: U[i*m+j]=1, V[j*p+4]=1, W[4*n+i]=1
    for i in range(4):
        for j in range(5):
            U[i * m + j][col] = 1
    for j in range(5):
        V[j * p + 4][col] = 1
    for i in range(4):
        W[4 * n + i][col] = 1
    col += 1
    
    # Correction term for overlap at (i,4,4)
    # T[i,4,4] got 1 from j=4 term and 1 from k=4 term. Need -1.
    # Rank-1: U[i*m+4]=-1, V[4*p+4]=1, W[4*n+i]=1 => -1*1*1 = -1
    for i in range(4):
        U[i * m + 4][col] += -1
    V[4 * p + 4][col] = 1
    for i in range(4):
        W[4 * n + i][col] = 1
    col += 1
    
    # Trim
    if col < rank_estimate:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
