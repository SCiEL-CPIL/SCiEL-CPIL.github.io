import time
import sys

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 3 Strassen blocks for A_L B_L (2x2x6 -> 3x2x2x2)
    # 3 Strassen blocks for A_R1 B_R1 (2x2x6 -> 3x2x2x2)
    # 12 simple terms for A_R2 B_R2 (2x1x6)
    rank = 3 * 7 + 3 * 7 + 12
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i1*m + j1][col] += 1
        U[i2*m + j2][col] += 1
        V[j1*p + k1][col] += 1
        V[j2*p + k2][col] += 1
        W[k1*n + i1][col] += 1
        W[k2*n + i2][col] += 1
        col += 1
        
        # M2: (A21+A22)B11
        U[i2*m + j1][col] += 1
        U[i2*m + j2][col] += 1
        V[j1*p + k1][col] += 1
        W[k1*n + i2][col] += 1
        W[k2*n + i2][col] += -1
        col += 1
        
        # M3: A11(B21-B22)
        U[i1*m + j1][col] += 1
        V[j2*p + k1][col] += 1
        V[j2*p + k2][col] += -1
        W[k2*n + i1][col] += 1
        W[k2*n + i2][col] += 1
        col += 1
        
        # M4: A22(B12-B11)
        U[i2*m + j2][col] += 1
        V[j1*p + k2][col] += 1
        V[j1*p + k1][col] += -1
        W[k1*n + i1][col] += 1
        W[k1*n + i2][col] += 1
        col += 1
        
        # M5: (A11+A12)B22
        U[i1*m + j1][col] += 1
        U[i1*m + j2][col] += 1
        V[j2*p + k2][col] += 1
        W[k1*n + i1][col] += -1
        W[k2*n + i1][col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[i2*m + j1][col] += 1
        U[i1*m + j1][col] += -1
        V[j1*p + k1][col] += 1
        V[j1*p + k2][col] += 1
        W[k2*n + i2][col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[i1*m + j2][col] += 1
        U[i2*m + j2][col] += -1
        V[j2*p + k1][col] += 1
        V[j2*p + k2][col] += 1
        W[k1*n + i1][col] += 1
        col += 1

    # Part 1: A_L B_L
    # A_L is rows 0,1 of A, cols 0,1
    # B_L is rows 0,1 of B, cols 0..5
    # Strassen blocks on cols (0,1), (2,3), (4,5)
    add_strassen_block(0, 1, 0, 1, 0, 1)
    add_strassen_block(0, 1, 0, 1, 2, 3)
    add_strassen_block(0, 1, 0, 1, 4, 5)
    
    # Part 2: A_R1 B_R1
    # A_R1 is rows 0,1 of A, cols 2,3
    # B_R1 is rows 2,3 of B, cols 0..5
    # Strassen blocks on cols (0,1), (2,3), (4,5)
    add_strassen_block(0, 1, 2, 3, 0, 1)
    add_strassen_block(0, 1, 2, 3, 2, 3)
    add_strassen_block(0, 1, 2, 3, 4, 5)
    
    # Part 3: A_R2 B_R2
    # A_R2 is rows 0,1 of A, col 4
    # B_R2 is row 4 of B, cols 0..5
    # Simple terms
    for i in range(n):
        for k in range(p):
            U[i*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + i][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
