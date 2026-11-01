# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=5
    # Rank 19 decomposition: Strassen for first 2 cols of B, Schoolbook for rest.
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # A is 2x2: indices 0,1,2,3 (row-major: 00, 01, 10, 11)
    # B is 2x5: indices 0..9 (row-major: 00,01,02,03,04, 10,11,12,13,14)
    # C is 5x2: indices 0..9 (row-major: 00,01, 10,11, 20,21, 30,31, 40,41)
    # Wait, W shape is (p*n, rank). p=5, n=2.
    # W[k*n+i, r]. k is col of B, i is row of A/C.
    # So W index is k*2 + i.
    
    col = 0
    
    # Strassen for B cols 0, 1
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[1][col] = 1 # C00, C10? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's map W indices carefully.
    # W[k*n + i]
    # k=0 (B col 0): W[0] is C[0,0], W[1] is C[1,0]
    # k=1 (B col 1): W[2] is C[0,1], W[3] is C[1,1]
    
    # M1
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1 # C00
    W[3][col] = 1 # C11
    col += 1
    
    # M2 = (A10+A11)B00
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[1][col] = 1 # C10
    W[3][col] = -1 # C11
    col += 1
    
    # M3 = A00(B10-B11)
    U[0][col] = 1
    V[5][col] = 1; V[6][col] = -1
    W[2][col] = 1 # C01
    W[3][col] = 1 # C11
    col += 1
    
    # M4 = A11(B01-B00)
    U[3][col] = 1
    V[1][col] = 1; V[0][col] = -1
    W[0][col] = 1 # C00
    W[1][col] = 1 # C10
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1 # C00
    W[2][col] = 1 # C01
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1 # C11
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[3][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1 # C00
    col += 1
    
    # Schoolbook for B cols 2, 3, 4
    for k in range(2, p):
        # C0k = A00*B0k + A01*B1k
        # C1k = A10*B0k + A11*B1k
        
        # T1 = A00*B0k
        U[0][col] = 1
        V[k*2][col] = 1 # B row 0, col k
        W[k*2][col] = 1 # C row 0, col k
        col += 1
        
        # T2 = A01*B1k
        U[1][col] = 1
        V[k*2+1][col] = 1 # B row 1, col k
        W[k*2][col] = 1 # C row 0, col k
        col += 1
        
        # T3 = A10*B0k
        U[2][col] = 1
        V[k*2][col] = 1
        W[k*2+1][col] = 1 # C row 1, col k
        col += 1
        
        # T4 = A11*B1k
        U[3][col] = 1
        V[k*2+1][col] = 1
        W[k*2+1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
