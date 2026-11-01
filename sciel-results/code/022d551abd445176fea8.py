import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    # We use the block Strassen decomposition.
    # p=16, so 8 blocks of size 2.
    # Each block uses rank 7.
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 4 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 32 x 56
    
    # Strassen terms for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Coefficients for A (a,b,c,d) indices 0,1,2,3
    # Coefficients for B (e,f,g,h) indices 0,1,2,3
    # Coefficients for C (C00, C01, C10, C11) indices 0,1,2,3
    
    # M1: A: [1, 0, 0, 1], B: [1, 0, 0, 1], C: [1, 0, 0, 1] (C00 +1, C11 +1)
    # M2: A: [0, 0, 1, 1], B: [1, 0, 0, 0], C: [0, 0, 1, 0] (C10 +1)
    # M3: A: [1, 0, 0, 0], B: [0, 1, 0, -1], C: [0, 1, 0, 0] (C01 +1)
    # M4: A: [0, 0, 0, 1], B: [0, 0, 1, -1], C: [1, 0, 1, 0] (C00 +1, C10 +1)
    # M5: A: [1, 1, 0, 0], B: [0, 0, 0, 1], C: [-1, 1, 0, 0] (C00 -1, C01 +1)
    # M6: A: [0, 0, 1, -1], B: [1, 1, 0, 0], C: [0, 0, 0, 1] (C11 +1)
    # M7: A: [0, 1, 0, -1], B: [0, 0, 1, 1], C: [1, 0, 0, 0] (C00 +1)
    
    strassen_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    
    strassen_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    strassen_C = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2
    for k in range(num_blocks):
        for t in range(7):
            col = k * 7 + t
            # Fill U
            for i in range(4):
                U[i][col] = strassen_A[t][i]
            # Fill V
            # B indices for block k are 4k to 4k+3
            for j in range(4):
                V[4*k + j][col] = strassen_B[t][j]
            # Fill W
            # C indices for block k are 4k to 4k+3
            for l in range(4):
                W[4*k + l][col] = strassen_C[t][l]
                
    return {"U": U, "V": V, "W": W}
