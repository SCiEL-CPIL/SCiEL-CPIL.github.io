import numpy as np
import time

def construct():
    n, m, p = 2, 2, 7
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 factors
    # U_strassen[i*2+j, col]
    U_s = [
        [1, 0, 0, 1, 1, -1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, -1]
    ]
    # V_strassen[j*2+k, col]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [1, 0, -1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    # W_strassen[k*2+i, col]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, -1, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0] # Wait, let's re-verify Strassen W
    ]
    
    # Let's use the standard Strassen W from literature to be safe.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U (A coefficients)
    # M1: A00=1, A11=1
    # M2: A10=1, A11=1
    # M3: A00=1
    # M4: A11=1
    # M5: A00=1, A01=1
    # M6: A10=1, A00=-1
    # M7: A01=1, A11=-1
    
    U_s = [
        [1, 0, 1, 0, 1, -1, 0], # A00 (i=0, j=0)
        [0, 0, 0, 0, 1, 0, 1],  # A01 (i=0, j=1)
        [0, 1, 0, 0, 0, 1, 0],  # A10 (i=1, j=0)
        [1, 1, 0, 1, 0, 0, -1]  # A11 (i=1, j=1)
    ]
    
    # V (B coefficients)
    # M1: B00=1, B11=1
    # M2: B00=1
    # M3: B10=1, B11=-1
    # M4: B01=1, B00=-1
    # M5: B11=1
    # M6: B00=1, B01=1
    # M7: B10=1, B11=1
    
    V_s = [
        [1, 1, 0, -1, 0, 1, 0], # B00 (j=0, k=0)
        [0, 0, 0, 1, 0, 1, 0],  # B01 (j=0, k=1)
        [0, 0, 1, 0, 0, 0, 1],  # B10 (j=1, k=0)
        [1, 0, -1, 0, 1, 0, 1]  # B11 (j=1, k=1)
    ]
    
    # W (C coefficients)
    # C00: M1=1, M4=1, M5=-1, M7=1
    # C01: M3=1, M5=1
    # C10: M2=1, M4=1
    # C11: M1=1, M2=-1, M3=1, M6=1
    
    W_s = [
        [1, 0, 0, 1, -1, 0, 1], # C00 (k=0, i=0)
        [0, 0, 1, 0, 1, 0, 0],  # C01 (k=1, i=0)
        [0, 1, 0, 1, 0, 0, 0],  # C10 (k=0, i=1)
        [1, -1, 1, 0, 0, 1, 0]  # C11 (k=1, i=1)
    ]
    
    # Block 1: k=0,1
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*2+j][r] = U_s[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*7+k][r] = V_s[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*2+i][r] = W_s[k*2+i][r]
                
    # Block 2: k=2,3
    # A is same, so U is same.
    # B is shifted, so V indices shift.
    # C is shifted, so W indices shift.
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*2+j][r+7] = U_s[i*2+j][r]
        for j in range(2):
            for k in range(2):
                # Local k is 0,1. Global k is 2,3.
                V[j*7+(k+2)][r+7] = V_s[j*2+k][r]
        for k in range(2):
            for i in range(2):
                # Local k is 0,1. Global k is 2,3.
                W[(k+2)*2+i][r+7] = W_s[k*2+i][r]
                
    # Block 3: k=4,5,6
    # Schoolbook
    col = 14
    for i in range(2):
        for j in range(2):
            for k in range(3):
                # Global k is 4,5,6
                gk = k + 4
                U[i*2+j][col] = 1.0
                V[j*7+gk][col] = 1.0
                W[gk*2+i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
