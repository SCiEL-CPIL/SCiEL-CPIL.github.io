import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 18
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 block (indices 0-6)
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    # C11 += M1, C22 += M1
    W[0][0] = 1  # C00 (k=0, i=0) -> idx 0
    W[3][0] = 1  # C11 (k=1, i=1) -> idx 1*3+1=4? 
    # Wait. W[k*n+i].
    # C00: k=0, i=0 -> 0
    # C11: k=1, i=1 -> 4
    # C01: k=1, i=0 -> 3
    # C10: k=0, i=1 -> 1
    
    # Let's map C indices to W indices:
    # C[i][k] -> W[k*n + i]
    # C00 -> W[0]
    # C01 -> W[3]
    # C02 -> W[6]
    # C10 -> W[1]
    # C11 -> W[4]
    # C12 -> W[7]
    # C20 -> W[2]
    # C21 -> W[5]
    # C22 -> W[8]
    
    # M1 contributes to C00 and C11
    W[0][0] = 1
    W[4][0] = 1
    
    # M2 = (A21+A22)B11
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    # C21 += M2 (C[2][1] -> W[1*3+2]=5)
    # C11? No.
    # Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # Note: My indices 0,1,2 correspond to rows/cols 1,2,3 in 1-based.
    # C00 is 1,1. C01 is 1,2. C10 is 2,1. C11 is 2,2.
    
    # M2 contributes to C10 (2,1) and C22? No.
    # C21 (2,1) -> W[1*3+1] = 4? No.
    # C[i][k]. i=1 (row 2), k=0 (col 1).
    # W[0*3+1] = 1.
    # So M2 contributes to W[1].
    # And C22 (2,2)? No, C22 is M1 - M2 + ...
    # So M2 contributes to C10 (W[1]) and C22 (W[4])?
    # C22 is i=1, k=1. W[1*3+1] = 4.
    # So M2 contributes to W[1] and W[4].
    # Wait, Strassen C22 = M1 - M2 + M3 + M6.
    # So M2 has coefficient -1 in C22.
    # And C21 = M2 + M4.
    # So M2 has coefficient 1 in C21.
    
    # Let's redo the mapping carefully.
    
    # Clear W for term 0
    W[0][0] = 1
    W[4][0] = 1
    
    # Term 1: M2
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    # C21 (i=1, k=0) -> W[0*3+1] = 1. Coeff 1.
    # C22 (i=1, k=1) -> W[1*3+1] = 4. Coeff -1.
    W[1][1] = 1
    W[4][1] = -1
    
    # Term 2: M3 = A11(B12-B22)
    U[0][2] = 1
    V[4][2] = -1; V[8][2] = 1 # B12 is V[4], B22 is V[8]
    # C12 (i=0, k=1) -> W[1*3+0] = 3. Coeff 1.
    # C22 (i=1, k=1) -> W[4]. Coeff 1.
    W[3][2] = 1
    W[4][2] = 1
    
    # Term 3: M4 = A22(B21-B11)
    U[4][3] = 1
    V[1][3] = 1; V[0][3] = -1 # B21 is V[1], B11 is V[0]
    # C11 (i=0, k=0) -> W[0]. Coeff 1.
    # C21 (i=1, k=0) -> W[1]. Coeff 1.
    W[0][3] = 1
    W[1][3] = 1
    
    # Term 4: M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[8][4] = 1
    # C11 (i=0, k=0) -> W[0]. Coeff -1.
    # C12 (i=0, k=1) -> W[3]. Coeff 1.
    W[0][4] = -1
    W[3][4] = 1
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[4][5] = 1
    # C22 (i=1, k=1) -> W[4]. Coeff 1.
    W[4][5] = 1
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[4][6] = -1
    V[1][6] = 1; V[8][6] = 1
    # C11 (i=0, k=0) -> W[0]. Coeff 1.
    W[0][6] = 1
    
    # Cross terms
    # Term 7: A11 B13
    U[0][7] = 1
    V[6][7] = 1 # B13 is V[1*3+0]=3? No.
    # V index is j*p+k.
    # B13: j=0, k=2. V[0*3+2] = 2.
    # B23: j=1, k=2. V[1*3+2] = 5.
    # B31: j=2, k=0. V[2*3+0] = 6.
    # B32: j=2, k=1. V[2*3+1] = 7.
    # B33: j=2, k=2. V[2*3+2] = 8.
    
    # Correction on V indices:
    # B00: 0
    # B01: 1
    # B02: 2
    # B10: 3
    # B11: 4
    # B12: 5
    # B20: 6
    # B21: 7
    # B22: 8
    
    # My previous V assignments in Strassen were:
    # M1: B11, B22 -> V[4], V[8]. Correct.
    # M2: B11 -> V[4]. Correct.
    # M3: B12, B22 -> V[5], V[8]. I used V[4], V[8] in code above?
    # In code above: V[4][2]=-1, V[8][2]=1.
    # B12 is V[5]. So I should use V[5].
    # Let's fix V indices.
    
    # Restart V assignments for Strassen
    # M1: B11(V4), B22(V8)
    # M2: B11(V4)
    # M3: B12(V5), B22(V8)
    # M4: B21(V7), B11(V4)
    # M5: B22(V8)
    # M6: B11(V4), B12(V5)
    # M7: B21(V7), B22(V8)
    
    # Let's rebuild U, V, W from scratch to be safe.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Term 0: M1
    U[0][0] = 1; U[4][0] = 1
    V[4][0] = 1; V[8][0] = 1
    W[0][0] = 1 # C00
    W[4][0] = 1 # C11
    
    # Term 1: M2
    U[3][1] = 1; U[4][1] = 1
    V[4][1] = 1
    W[1][1] = 1 # C10
    W[4][1] = -1 # C11
    
    # Term 2: M3
    U[0][2] = 1
    V[5][2] = 1; V[8][2] = -1 # B12 - B22
    W[3][2] = 1 # C01
    W[4][2] = 1 # C11
    
    # Term 3: M4
    U[4][3] = 1
    V[7][3] = 1; V[4][3] = -1 # B21 - B11
    W[0][3] = 1 # C00
    W[1][3] = 1 # C10
    
    # Term 4: M5
    U[0][4] = 1; U[1][4] = 1
    V[8][4] = 1
    W[0][4] = -1 # C00
    W[3][4] = 1 # C01
    
    # Term 5: M6
    U[3][5] = 1; U[0][5] = -1
    V[4][5] = 1; V[5][5] = 1
    W[4][5] = 1 # C11
    
    # Term 6: M7
    U[1][6] = 1; U[4][6] = -1
    V[7][6] = 1; V[8][6] = 1
    W[0][6] = 1 # C00
    
    # Term 7: A00 B02 (A11 B13)
    U[0][7] = 1
    V[2][7] = 1
    W[6][7] = 1 # C02 (i=0, k=2) -> W[2*3+0]=6
    
    # Term 8: A01 B12 (A12 B23)
    U[1][8] = 1
    V[5][8] = 1
    W[6][8] = 1 # C02
    
    # Term 9: A10 B02 (A21 B13)
    U[3][9] = 1
    V[2][9] = 1
    W[7][9] = 1 # C12 (i=1, k=2) -> W[2*3+1]=7
    
    # Term 10: A11 B12 (A22 B23)
    U[4][10] = 1
    V[5][10] = 1
    W[7][10] = 1 # C12
    
    # Term 11: A20 B00 (A31 B11)
    U[6][11] = 1
    V[4][11] = 1
    W[2][11] = 1 # C20 (i=2, k=0) -> W[0*3+2]=2
    
    # Term 12: A21 B10 (A32 B21)
    U[7][12] = 1
    V[7][12] = 1
    W[2][12] = 1 # C20
    
    # Term 13: A20 B01 (A31 B12)
    U[6][13] = 1
    V[5][13] = 1
    W[5][13] = 1 # C21 (i=2, k=1) -> W[1*3+2]=5
    
    # Term 14: A21 B11 (A32 B22)
    U[7][14] = 1
    V[8][14] = 1
    W[5][14] = 1 # C21
    
    # Term 15: A20 B20 (A31 B31)
    U[6][15] = 1
    V[6][15] = 1
    W[8][15] = 1 # C22 (i=2, k=2) -> W[2*3+2]=8
    
    # Term 16: A21 B21 (A32 B32)
    U[7][16] = 1
    V[7][16] = 1
    W[8][16] = 1 # C22
    
    # Term 17: A22 B22 (A33 B33)
    U[8][17] = 1
    V[8][17] = 1
    W[8][17] = 1 # C22
    
    return {"U": U, "V": V, "W": W}
