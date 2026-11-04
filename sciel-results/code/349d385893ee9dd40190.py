import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    # A is 2x3, B is 3x5
    # Split A into A1 (2x2) and A2 (2x1)
    # Split B into B1 (3x2) and B2 (3x3)
    # C = A1 B1 + A1 B2 + A2 B1 + A2 B2
    
    # A1 B1: 2x2 by 2x2 (Rank 7)
    # A2 B1: 2x1 by 2x2 (Rank 4)
    # A1 B2: 2x2 by 2x3 (Rank 11)
    # A2 B2: 2x1 by 2x3 (Rank 6)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A1 B1 (2x2 by 2x2)
    # A1 indices: i in 0..1, j in 0..1
    # B1 indices: j in 0..1, k in 0..1
    # Strassen terms
    # M1 = (A11 + A22)(B11 + B22)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[2][col] = 1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # M2 = (A21 + A22) B11
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[1][col] = 1
    col += 1
    
    # M3 = A11 (B21 - B22)
    U[0][col] = 1
    V[1][col] = 1; V[2][col] = -1
    W[0][col] = 1
    col += 1
    
    # M4 = A22 (B11 - B21)
    U[3][col] = 1
    V[0][col] = 1; V[1][col] = -1
    W[1][col] = 1
    col += 1
    
    # M5 = (A11 + A12) B22
    U[0][col] = 1; U[1][col] = 1
    V[2][col] = 1
    W[0][col] = 1
    col += 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[1][col] = 1
    col += 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1][col] = 1; U[3][col] = -1
    V[1][col] = 1; V[2][col] = 1
    W[0][col] = 1
    col += 1
    
    # Block 2: A2 B1 (2x1 by 2x2)
    # A2 indices: i in 0..1, j in 2
    # B1 indices: j in 0..1, k in 0..1
    # Schoolbook
    # A20 B100
    U[2][col] = 1
    V[0][col] = 1
    W[0][col] = 1
    col += 1
    # A20 B110
    U[2][col] = 1
    V[1][col] = 1
    W[0][col] = 1
    col += 1
    # A21 B101
    U[5][col] = 1
    V[0][col] = 1
    W[1][col] = 1
    col += 1
    # A21 B111
    U[5][col] = 1
    V[1][col] = 1
    W[1][col] = 1
    col += 1
    
    # Block 3: A1 B2 (2x2 by 2x3)
    # A1 indices: i in 0..1, j in 0..1
    # B2 indices: j in 0..1, k in 2..4
    # Rank 11: Strassen for 2x2 by 2x2 part, Schoolbook for 2x2 by 2x1 part
    # B2_11: j in 0..1, k in 2..3 (2x2)
    # B2_12: j in 0..1, k in 4 (2x1)
    
    # Strassen for A1 B2_11
    # M1 = (A11 + A22)(B211 + B222)
    U[0][col] = 1; U[3][col] = 1
    V[2][col] = 1; V[3][col] = 1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = (A21 + A22) B211
    U[2][col] = 1; U[3][col] = 1
    V[2][col] = 1
    W[3][col] = 1
    col += 1
    
    # M3 = A11 (B221 - B222)
    U[0][col] = 1
    V[3][col] = 1; V[4][col] = -1
    W[2][col] = 1
    col += 1
    
    # M4 = A22 (B211 - B221)
    U[3][col] = 1
    V[2][col] = 1; V[3][col] = -1
    W[3][col] = 1
    col += 1
    
    # M5 = (A11 + A12) B222
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    # M6 = (A21 - A11)(B211 + B212)
    U[2][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7 = (A12 - A22)(B221 + B222)
    U[1][col] = 1; U[3][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    # Schoolbook for A1 B2_12 (2x2 by 2x1)
    # A10 B202
    U[0][col] = 1
    V[4][col] = 1
    W[4][col] = 1
    col += 1
    # A11 B212
    U[1][col] = 1
    V[4][col] = 1
    W[4][col] = 1
    col += 1
    # A20 B203
    U[2][col] = 1
    V[4][col] = 1
    W[4][col] = 1
    col += 1
    # A21 B213
    U[3][col] = 1
    V[4][col] = 1
    W[4][col] = 1
    col += 1
    
    # Block 4: A2 B2 (2x1 by 2x3)
    # A2 indices: i in 0..1, j in 2
    # B2 indices: j in 0..1, k in 2..4
    # Schoolbook
    # A20 B202
    U[2][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    # A20 B212
    U[2][col] = 1
    V[4][col] = 1
    W[3][col] = 1
    col += 1
    # A20 B222
    U[2][col] = 1
    V[4][col] = 1
    W[4][col] = 1
    col += 1
    # A21 B203
    U[5][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    # A21 B213
    U[5][col] = 1
    V[4][col] = 1
    W[3][col] = 1
    col += 1
    # A21 B223
    U[5][col] = 1
    V[4][col] = 1
    W[4][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
