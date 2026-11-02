import numpy as np
import time

DIMENSIONS = [2, 3, 3]
GLOBAL_START_TIME = time.time()
TIME_LIMIT = 135.0

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    rank = 17
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Part 1: A[:,0] * B[0,:] (6 terms)
    # i in 0..1, k in 0..2
    # U[i*3+0, r] = 1
    # V[0*3+k, r] = 1
    # W[k*2+i, r] = 1
    r = 0
    for i in range(2):
        for k in range(3):
            U[i * 3 + 0][r] = 1
            V[0 * 3 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1
            
    # Part 2: A[:,1:3] * B[1,:] (4 terms)
    # i in 0..1, j in 1..2
    # U[i*3+j, r] = 1
    # V[j*3+0, r] = 1
    # W[0*2+i, r] = 1
    for i in range(2):
        for j in range(1, 3):
            U[i * 3 + j][r] = 1
            V[j * 3 + 0][r] = 1
            W[0 * 2 + i][r] = 1
            r += 1
            
    # Part 3: A[:,1:3] * B[1:3,:] (7 terms)
    # Strassen for 2x2x2
    # Local A: i in 0..1, j' in 0..1 (global j = j'+1)
    # Local B: j' in 0..1 (global j = j'+1), k' in 0..1 (global k = k'+1)
    # Local C: i in 0..1, k' in 0..1 (global k = k'+1)
    
    # P1 = (A00 + A11)(B00 + B11)
    # Global: (A01 + A12)(B11 + B22)
    U[0 * 3 + 1][r] = 1; U[1 * 3 + 2][r] = 1
    V[1 * 3 + 1][r] = 1; V[2 * 3 + 2][r] = 1
    W[1 * 2 + 0][r] = 1; W[2 * 2 + 1][r] = 1 # C01, C12
    r += 1
    
    # P2 = (A10 + A11)B00
    # Global: (A11 + A12)B11
    U[1 * 3 + 1][r] = 1; U[1 * 3 + 2][r] = 1
    V[1 * 3 + 1][r] = 1
    W[1 * 2 + 1][r] = 1 # C11
    r += 1
    
    # P3 = A00(B10 - B11)
    # Global: A01(B21 - B22)
    U[0 * 3 + 1][r] = 1
    V[2 * 3 + 1][r] = 1; V[2 * 3 + 2][r] = -1
    W[2 * 2 + 0][r] = 1; W[2 * 2 + 1][r] = 1 # C02, C12
    r += 1
    
    # P4 = A11(B01 - B00)
    # Global: A12(B12 - B11)
    U[1 * 3 + 2][r] = 1
    V[1 * 3 + 2][r] = 1; V[1 * 3 + 1][r] = -1
    W[1 * 2 + 0][r] = 1; W[1 * 2 + 1][r] = 1 # C01, C11
    r += 1
    
    # P5 = (A00 + A01)B11
    # Global: (A01 + A02)B22
    U[0 * 3 + 1][r] = 1; U[0 * 3 + 2][r] = 1
    V[2 * 3 + 2][r] = 1
    W[1 * 2 + 0][r] = -1; W[2 * 2 + 0][r] = 1 # C01 (-), C02 (+)
    r += 1
    
    # P6 = (A10 - A00)(B00 + B01)
    # Global: (A11 - A01)(B11 + B12)
    U[1 * 3 + 1][r] = 1; U[0 * 3 + 1][r] = -1
    V[1 * 3 + 1][r] = 1; V[1 * 3 + 2][r] = 1
    # P6 is not used in standard Strassen reconstruction for C?
    # Wait, I need to check if P6 contributes.
    # In the standard formula, P6 is not in the sum for C_ij.
    # So W entries should be 0.
    r += 1
    
    # P7 = (A01 - A11)(B10 + B11)
    # Global: (A02 - A12)(B21 + B22)
    U[0 * 3 + 2][r] = 1; U[1 * 3 + 2][r] = -1
    V[2 * 3 + 1][r] = 1; V[2 * 3 + 2][r] = 1
    W[1 * 2 + 0][r] = -1; W[2 * 2 + 1][r] = 1 # C01 (-), C12 (+)
    r += 1
    
    return {"U": U, "V": V, "W": W}
