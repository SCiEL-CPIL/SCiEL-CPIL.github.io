import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 4, 4, 7
    
    # Strassen factors for 2x2x2
    # U: (4, 7), V: (4, 7), W: (4, 7)
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11) -> C00+C11
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2: (A10+A11)B00 -> C10+C11
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = 1
    
    # M3: A00(B10-B11) -> C01+C11
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # M4: A11(B01-B00) -> C00+C10
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # M5: (A00+A10)B11 -> -C00+C01
    U2[0][4] = 1; U2[2][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # M6: (A11-A00)(B00+B10) -> C11
    U2[3][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[2][5] = 1
    W2[3][5] = 1
    
    # M7: (A01-A11)(B11+B01) -> -C00+C11
    U2[1][6] = 1; U2[3][6] = -1
    V2[3][6] = 1; V2[1][6] = 1
    W2[0][6] = -1; W2[3][6] = 1
    
    # Construct 4x4x4 factors (rank 49)
    rank_44 = 49
    U44 = [[0]*rank_44 for _ in range(16)]
    V44 = [[0]*rank_44 for _ in range(16)]
    W44 = [[0]*rank_44 for _ in range(16)]
    
    for i in range(4):
        i1, i2 = divmod(i, 2)
        for j in range(4):
            j1, j2 = divmod(j, 2)
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    U44[i*4+j][c] = U2[i1*2+j1][c1] * U2[i2*2+j2][c2]
    
    for j in range(4):
        j1, j2 = divmod(j, 2)
        for k in range(4):
            k1, k2 = divmod(k, 2)
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    V44[j*4+k][c] = V2[j1*2+k1][c1] * V2[j2*2+k2][c2]
    
    for k in range(4):
        k1, k2 = divmod(k, 2)
        for i in range(4):
            i1, i2 = divmod(i, 2)
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    W44[k*4+i][c] = W2[k1*2+i1][c1] * W2[k2*2+i2][c2]
                    
    # Total rank = 49 (Strassen 4x4x4) + 48 (Schoolbook 4x4x3)
    rank_total = 49 + 48
    
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(28)]
    W = [[0]*rank_total for _ in range(28)]
    
    # Fill in the 4x4x4 part (k=0..3)
    for i in range(4):
        for j in range(4):
            for c in range(49):
                U[i*4+j][c] = U44[i*4+j][c]
                
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*7+k][c] = V44[j*4+k][c]
                
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k*4+i][c] = W44[k*4+i][c]
                
    # Fill in the 4x4x3 part (k=4..6) with schoolbook
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(3):
                c = col_offset + (i*4 + j)*3 + k
                U[i*4+j][c] = 1
                V[j*7 + (k+4)][c] = 1
                W[(k+4)*4 + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
