import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for l in range(3):
        # Block l corresponds to B columns 3l, 3l+1, 3l+2
        # and C columns 3l, 3l+1, 3l+2
        
        # 1. Strassen terms (7)
        # M1
        U[0*3+0][col] += 1 # a00
        U[1*3+1][col] += 1 # a11
        V[0*9+3*l][col] += 1 # b0, 3l
        V[1*9+3*l+1][col] += 1 # b1, 3l+1
        W[(3*l+0)*2+0][col] += 1 # C0, 3l
        W[(3*l+0)*2+1][col] += 1 # C1, 3l
        W[(3*l+1)*2+0][col] += 1 # C0, 3l+1
        W[(3*l+1)*2+1][col] += 1 # C1, 3l+1
        # Wait, Strassen formulas:
        # C00 += M1 + M4 - M5 + M7
        # C01 += M3 - M4 + M5
        # C10 += M1 - M2 + M6 - M7
        # C11 += M2 - M3 + M6
        
        # Let's add coefficients to W for M1
        W[(3*l+0)*2+0][col] += 1
        W[(3*l+1)*2+0][col] += 1
        col += 1
        
        # M2
        U[1*3+0][col] += 1 # a10
        U[1*3+1][col] += 1 # a11
        V[0*9+3*l][col] += 1 # b0, 3l
        W[(3*l+0)*2+1][col] += -1 # C10 -= M2
        W[(3*l+1)*2+1][col] += 1 # C11 += M2
        col += 1
        
        # M3
        U[0*3+0][col] += 1 # a00
        V[0*9+3*l+1][col] += 1 # b0, 3l+1
        V[1*9+3*l+1][col] += -1 # b1, 3l+1
        W[(3*l+1)*2+0][col] += 1 # C01 += M3
        W[(3*l+1)*2+1][col] += -1 # C11 -= M3
        col += 1
        
        # M4
        U[1*3+1][col] += 1 # a11
        V[1*9+3*l][col] += 1 # b1, 3l
        V[0*9+3*l][col] += -1 # b0, 3l
        W[(3*l+0)*2+0][col] += 1 # C00 += M4
        W[(3*l+1)*2+0][col] += -1 # C01 -= M4
        col += 1
        
        # M5
        U[0*3+0][col] += 1 # a00
        U[0*3+1][col] += 1 # a01
        V[1*9+3*l+1][col] += 1 # b1, 3l+1
        W[(3*l+0)*2+0][col] += -1 # C00 -= M5
        W[(3*l+1)*2+0][col] += 1 # C01 += M5
        col += 1
        
        # M6
        U[1*3+0][col] += 1 # a10
        U[0*3+0][col] += -1 # a00
        V[0*9+3*l][col] += 1 # b0, 3l
        V[0*9+3*l+1][col] += 1 # b0, 3l+1
        W[(3*l+0)*2+1][col] += 1 # C10 += M6
        W[(3*l+1)*2+1][col] += 1 # C11 += M6
        col += 1
        
        # M7
        U[0*3+1][col] += 1 # a01
        U[1*3+1][col] += -1 # a11
        V[1*9+3*l][col] += 1 # b1, 3l
        V[1*9+3*l+1][col] += 1 # b1, 3l+1
        W[(3*l+0)*2+0][col] += 1 # C00 += M7
        W[(3*l+0)*2+1][col] += -1 # C10 -= M7
        col += 1
        
        # 2. A2x2 B2x1 terms (4)
        # P1 = a00 b0, 3l+2
        U[0*3+0][col] += 1
        V[0*9+3*l+2][col] += 1
        W[(3*l+2)*2+0][col] += 1 # C0, 3l+2
        col += 1
        
        # P2 = a01 b1, 3l+2
        U[0*3+1][col] += 1
        V[1*9+3*l+2][col] += 1
        W[(3*l+2)*2+0][col] += 1 # C0, 3l+2
        col += 1
        
        # P3 = a10 b0, 3l+2
        U[1*3+0][col] += 1
        V[0*9+3*l+2][col] += 1
        W[(3*l+2)*2+1][col] += 1 # C1, 3l+2
        col += 1
        
        # P4 = a11 b1, 3l+2
        U[1*3+1][col] += 1
        V[1*9+3*l+2][col] += 1
        W[(3*l+2)*2+1][col] += 1 # C1, 3l+2
        col += 1
        
        # 3. Outer product term (1)
        # Q = a_i, 2 * b_2, 3l+k
        # U has 1s at (0,2) and (1,2)
        U[0*3+2][col] += 1
        U[1*3+2][col] += 1
        # V has 1s at (2, 3l), (2, 3l+1), (2, 3l+2)
        V[2*9+3*l][col] += 1
        V[2*9+3*l+1][col] += 1
        V[2*9+3*l+2][col] += 1
        # W has 1s at all (i, 3l+k) for i=0,1 and k=0,1,2
        for i in range(2):
            for k in range(3):
                W[(3*l+k)*2+i][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
