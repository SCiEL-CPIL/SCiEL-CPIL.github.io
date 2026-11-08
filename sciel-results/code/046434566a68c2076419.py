import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    # 5 blocks of 2x3x2, each rank 11
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 5 blocks of width 2 for p=10
    for block in range(5):
        k_start = block * 2
        k1 = k_start
        k2 = k_start + 1
        
        # Strassen part for A[:, 0:2] and B[0:2, k1:k2]
        # A indices: i*m + j. i in 0..1, j in 0..1
        # B indices: j*p + k. j in 0..1, k in k1..k2
        
        # M1 = (A00+A11)(B00+B11)
        U[0*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*10+k1][col] = 1
        V[1*10+k2][col] = 1
        W[k1*2+0][col] = 1
        W[k2*2+1][col] = 1
        W[k1*2+1][col] = -1
        W[k2*2+0][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[1*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*10+k1][col] = 1
        W[k1*2+1][col] = 1
        W[k2*2+1][col] = -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[0*3+0][col] = 1
        V[0*10+k2][col] = 1
        V[1*10+k2][col] = -1
        W[k1*2+0][col] = 1
        W[k2*2+0][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[1*3+1][col] = 1
        V[1*10+k1][col] = 1
        V[0*10+k1][col] = -1
        W[k1*2+0][col] = 1
        W[k2*2+1][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[0*3+0][col] = 1
        U[0*3+1][col] = 1
        V[1*10+k2][col] = 1
        W[k1*2+0][col] = -1
        W[k2*2+1][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1*3+0][col] = 1
        U[0*3+0][col] = -1
        V[0*10+k1][col] = 1
        V[0*10+k2][col] = 1
        W[k2*2+1][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0*3+1][col] = 1
        U[1*3+1][col] = -1
        V[1*10+k1][col] = 1
        V[1*10+k2][col] = 1
        W[k1*2+0][col] = 1
        W[k2*2+1][col] = -1
        col += 1
        
        # Outer product part for A[:, 2] and B[2, k1:k2]
        # A index: i*3+2. B index: 2*10+k.
        # C11 += A02 * B20
        U[0*3+2][col] = 1
        V[2*10+k1][col] = 1
        W[k1*2+0][col] = 1
        col += 1
        
        # C12 += A02 * B21
        U[0*3+2][col] = 1
        V[2*10+k2][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # C21 += A12 * B20
        U[1*3+2][col] = 1
        V[2*10+k1][col] = 1
        W[k2*2+0][col] = 1
        col += 1
        
        # C22 += A12 * B21
        U[1*3+2][col] = 1
        V[2*10+k2][col] = 1
        W[k2*2+1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
