import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B11)
    # M4 = A11 (B01 + B11)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1
        U[0][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        V[8+k1][col] += 1
        W[3*k0][col] += 1
        W[3*k1][col] += 1
        W[3*k0+1][col] += 1
        W[3*k1+1][col] += 1
        col += 1
        
        # M2
        U[3][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        W[3*k0+1][col] += 1
        W[3*k1+1][col] += 1
        col += 1
        
        # M3
        U[0][col] += 1
        V[8+k0][col] += 1
        V[8+k1][col] -= 1
        W[3*k0][col] += 1
        W[3*k1][col] += 1
        col += 1
        
        # M4
        U[4][col] += 1
        V[k1][col] += 1
        V[8+k1][col] += 1
        W[3*k0][col] += 1
        W[3*k1][col] += 1
        W[3*k0+1][col] += 1
        W[3*k1+1][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1
        U[1][col] += 1
        V[8+k1][col] += 1
        W[3*k0][col] -= 1
        W[3*k1][col] -= 1
        W[3*k0+1][col] += 1
        W[3*k1+1][col] += 1
        col += 1
        
        # M6
        U[3][col] += 1
        U[0][col] -= 1
        V[k0][col] += 1
        V[k1][col] += 1
        W[3*k0+1][col] -= 1
        W[3*k1+1][col] -= 1
        col += 1
        
        # M7
        U[1][col] += 1
        U[4][col] -= 1
        V[8+k0][col] += 1
        V[8+k1][col] += 1
        W[3*k0][col] += 1
        W[3*k1][col] += 1
        col += 1

    # Part 2: A[0:2, 2] * B[2, 0:8]
    for i in range(2):
        for k in range(8):
            U[i*3+2][col] += 1
            V[16+k][col] += 1
            W[3*k+i][col] += 1
            col += 1

    # Part 3: A[2, 0:2] * B[0:2, 0:8]
    for j in range(2):
        for k in range(8):
            U[6+j][col] += 1
            V[j*8+k][col] += 1
            W[3*k+2][col] += 1
            col += 1

    # Part 4: A[2, 2] * B[2, 0:8]
    for k in range(8):
        U[8][col] += 1
        V[16+k][col] += 1
        W[3*k+2][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
