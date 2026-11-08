import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 39
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term: U[idx_u, col] += val_u, V[idx_v, col] += val_v, W[idx_w, col] += val_w
    # But Strassen is not just single terms.
    
    # Block 1: A1 (2x2) * B1 (2x7)
    # A1 rows: 0,1. Cols: 0,1.
    # B1 rows: 0,1. Cols: 0..6.
    
    # Sub-block 1.1: k=0,1. Strassen.
    # A indices: (0,0), (0,1), (1,0), (1,1) -> flat 0,1,2,3
    # B indices: (0,0), (0,1), (1,0), (1,1) -> flat 0,1,7,8
    # C indices: (0,0), (0,1), (1,0), (1,1) -> flat 0,1,2,3 (W is k*n+i)
    
    # Strassen 1
    # M1: (A00+A11)(B00+B11)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[8][col] = 1
    # C00 += M1, C11 += M1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2: (A10+A11)B00
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    # C10 += M2
    W[2][col] = 1
    col += 1
    
    # M3: A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[8][col] = -1
    # C01 += M3
    W[1][col] = 1
    col += 1
    
    # M4: A11(B10-B00)
    U[3][col] = 1
    V[7][col] = 1; V[0][col] = -1
    # C00 += M4, C10 += M4
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # M5: (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1
    # C00 -= M5, C01 += M5
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6: (A10-A00)(B00+B01)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    # C11 += M6
    W[3][col] = 1
    col += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][col] = 1; U[3][col] = -1
    V[7][col] = 1; V[8][col] = 1
    # C00 += M7
    W[0][col] = 1
    col += 1
    
    # Sub-block 1.2: k=2,3. Strassen.
    # B cols 2,3. Indices: (0,2)=2, (0,3)=3, (1,2)=9, (1,3)=10
    # C cols 2,3. Indices: (0,2)=2, (0,3)=3, (1,2)=4, (1,3)=5
    
    # M1
    U[0][col] = 1; U[3][col] = 1
    V[2][col] = 1; V[10][col] = 1
    W[2][col] = 1; W[5][col] = 1
    col += 1
    # M2
    U[2][col] = 1; U[3][col] = 1
    V[2][col] = 1
    W[4][col] = 1
    col += 1
    # M3
    U[0][col] = 1
    V[3][col] = 1; V[10][col] = -1
    W[3][col] = 1
    col += 1
    # M4
    U[3][col] = 1
    V[9][col] = 1; V[2][col] = -1
    W[2][col] = 1; W[4][col] = 1
    col += 1
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[10][col] = 1
    W[2][col] = -1; W[3][col] = 1
    col += 1
    # M6
    U[2][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[5][col] = 1
    col += 1
    # M7
    U[1][col] = 1; U[3][col] = -1
    V[9][col] = 1; V[10][col] = 1
    W[2][col] = 1
    col += 1
    
    # Sub-block 1.3: k=4,5. Strassen.
    # B cols 4,5. Indices: 4,5,11,12
    # C cols 4,5. Indices: 4,5,6,7
    
    # M1
    U[0][col] = 1; U[3][col] = 1
    V[4][col] = 1; V[12][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    # M2
    U[2][col] = 1; U[3][col] = 1
    V[4][col] = 1
    W[6][col] = 1
    col += 1
    # M3
    U[0][col] = 1
    V[5][col] = 1; V[12][col] = -1
    W[5][col] = 1
    col += 1
    # M4
    U[3][col] = 1
    V[11][col] = 1; V[4][col] = -1
    W[4][col] = 1; W[6][col] = 1
    col += 1
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[12][col] = 1
    W[4][col] = -1; W[5][col] = 1
    col += 1
    # M6
    U[2][col] = 1; U[0][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[7][col] = 1
    col += 1
    # M7
    U[1][col] = 1; U[3][col] = -1
    V[11][col] = 1; V[12][col] = 1
    W[4][col] = 1
    col += 1
    
    # Sub-block 1.4: k=6. Schoolbook 2x2 by 2x1.
    # B col 6. Indices: (0,6)=6, (1,6)=13
    # C col 6. Indices: (0,6)=6, (1,6)=7
    
    # C06 = A00 B06 + A01 B16
    U[0][col] = 1
    V[6][col] = 1
    W[6][col] = 1
    col += 1
    U[1][col] = 1
    V[13][col] = 1
    W[6][col] = 1
    col += 1
    # C16 = A10 B06 + A11 B16
    U[2][col] = 1
    V[6][col] = 1
    W[7][col] = 1
    col += 1
    U[3][col] = 1
    V[13][col] = 1
    W[7][col] = 1
    col += 1
    
    # Block 2: A2 (2x1) * B2 (1x7)
    # A2 rows 0,1. Col 2. Indices: (0,2)=4, (1,2)=5
    # B2 row 2. Cols 0..6. Indices: 14,15,16,17,18,19,20
    # C rows 0,1. Cols 0..6.
    
    for k in range(7):
        # C0k = A02 B2k
        U[4][col] = 1
        V[14 + k][col] = 1
        W[k][col] = 1 # C(0, k) -> index k
        col += 1
        # C1k = A12 B2k
        U[5][col] = 1
        V[14 + k][col] = 1
        W[7 + k][col] = 1 # C(1, k) -> index 7+k
        col += 1
        
    return {"U": U, "V": V, "W": W}
