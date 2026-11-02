import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 15 decomposition
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A01-A10)(B00+B10)
    # M7: (A11-A00)(B01+B11)
    
    # U, V, W for one 2x2x2 block
    # U[i, j, r]
    # V[j, k, r]
    # W[k, i, r]
    
    # r=0 (M1)
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*4+0][0] = 1; V[1*4+1][0] = 1
    W[0*2+0][0] = 1; W[1*2+1][0] = 1; W[0*2+1][0] = 1; W[1*2+0][0] = 1
    
    # r=1 (M2)
    U[1*3+0][1] = 1; U[1*3+1][1] = 1
    V[0*4+0][1] = 1
    W[0*2+1][1] = 1; W[1*2+1][1] = 1
    
    # r=2 (M3)
    U[0*3+0][2] = 1
    V[0*4+1][2] = 1; V[1*4+1][2] = -1
    W[1*2+0][2] = 1; W[1*2+1][2] = 1
    
    # r=3 (M4)
    U[1*3+1][3] = 1
    V[1*4+0][3] = 1; V[0*4+0][3] = -1
    W[0*2+0][3] = 1; W[0*2+1][3] = 1
    
    # r=4 (M5)
    U[0*3+0][4] = 1; U[0*3+1][4] = 1
    V[1*4+1][4] = 1
    W[0*2+0][4] = -1; W[1*2+0][4] = -1; W[0*2+1][4] = -1; W[1*2+1][4] = -1
    
    # r=5 (M6)
    U[0*3+1][5] = 1; U[1*3+0][5] = -1
    V[0*4+0][5] = 1; V[1*4+0][5] = 1
    W[1*2+0][5] = 1; W[1*2+1][5] = 1
    
    # r=6 (M7)
    U[1*3+1][6] = 1; U[0*3+0][6] = -1
    V[0*4+1][6] = 1; V[1*4+1][6] = 1
    W[0*2+0][6] = 1; W[0*2+1][6] = 1
    
    # Copy for r=7..13 (cols 2,3)
    for r in range(7, 14):
        src = r - 7
        for i in range(2):
            for j in range(2):
                U[i*3+j][r] = U[i*3+j][src]
        for j in range(2):
            for k in range(2):
                # Map k=0->2, k=1->3
                V[j*4+k+2][r] = V[j*4+k][src]
        for k in range(2):
            for i in range(2):
                W[k+2+2*i][r] = W[k+2*i][src]
                
    # r=14 (Remainder)
    for i in range(2):
        U[i*3+2][14] = 1
    for k in range(4):
        V[2*4+k][14] = 1
    for k in range(4):
        for i in range(2):
            W[k*2+i][14] = 1
            
    return {"U": U, "V": V, "W": W}
