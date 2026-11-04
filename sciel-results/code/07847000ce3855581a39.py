import time
import numpy as np

def construct():
    """
    Return the rank-17 decomposition of the 2x3 x 3x3 matrix
    multiplication tensor.
    """
    n, m, p = 2, 3, 3
    rank = 17
    
    # Initialize U, V, W
    U = [[0.0] * rank for _ in range(n * m)] # 6 x 17
    V = [[0.0] * rank for _ in range(m * p)] # 9 x 17
    W = [[0.0] * rank for _ in range(p * n)] # 6 x 17
    
    col = 0
    
    # --- Block 1: Strassen for A1 B1 (2x2 x 2x2) ---
    # A1 indices: 0, 1, 3, 4 (a11, a12, a21, a22)
    # B1 indices: 0, 1, 3, 4 (b11, b12, b21, b22)
    # C indices: 0, 1, 3, 4 (c11, c12, c21, c22)
    
    # M1: (a11+a22)(b11+b22)
    U[0][col] += 1; U[4][col] += 1
    V[0][col] += 1; V[4][col] += 1
    W[0][col] += 1; W[4][col] += 1
    col += 1
    
    # M2: (a21+a22)b11
    U[3][col] += 1; U[4][col] += 1
    V[0][col] += 1
    W[3][col] += 1; W[4][col] += -1
    col += 1
    
    # M3: a11(b12-b22)
    U[0][col] += 1
    V[1][col] += 1; V[4][col] += -1
    W[1][col] += 1; W[4][col] += 1
    col += 1
    
    # M4: a22(b21-b11)
    U[4][col] += 1
    V[3][col] += 1; V[0][col] += -1
    W[0][col] += 1; W[3][col] += 1
    col += 1
    
    # M5: (a11+a12)b22
    U[0][col] += 1; U[1][col] += 1
    V[4][col] += 1
    W[0][col] += -1; W[1][col] += 1
    col += 1
    
    # M6: (a21-a11)(b11+b12)
    U[3][col] += 1; U[0][col] += -1
    V[0][col] += 1; V[1][col] += 1
    W[4][col] += 1
    col += 1
    
    # M7: (a12-a22)(b21+b22)
    U[1][col] += 1; U[4][col] += -1
    V[3][col] += 1; V[4][col] += 1
    W[0][col] += 1
    col += 1
    
    # --- Block 2: Direct for A1 B2 (2x2 x 2x1) ---
    # A1 indices: 0, 1, 3, 4
    # B2 indices: 2, 5 (b13, b23)
    # C indices: 2, 5 (c13, c23)
    
    # a11 b13
    U[0][col] += 1
    V[2][col] += 1
    W[2][col] += 1
    col += 1
    
    # a12 b23
    U[1][col] += 1
    V[5][col] += 1
    W[2][col] += 1
    col += 1
    
    # a21 b13
    U[3][col] += 1
    V[2][col] += 1
    W[5][col] += 1
    col += 1
    
    # a22 b23
    U[4][col] += 1
    V[5][col] += 1
    W[5][col] += 1
    col += 1
    
    # --- Block 3: Direct for A2 B3 (2x1 x 1x3) ---
    # A2 indices: 2, 5 (a13, a23)
    # B3 indices: 6, 7, 8 (b31, b32, b33)
    # C indices: 0, 1, 2, 3, 4, 5
    
    # a13 b31
    U[2][col] += 1
    V[6][col] += 1
    W[0][col] += 1
    col += 1
    
    # a13 b32
    U[2][col] += 1
    V[7][col] += 1
    W[1][col] += 1
    col += 1
    
    # a13 b33
    U[2][col] += 1
    V[8][col] += 1
    W[2][col] += 1
    col += 1
    
    # a23 b31
    U[5][col] += 1
    V[6][col] += 1
    W[3][col] += 1
    col += 1
    
    # a23 b32
    U[5][col] += 1
    V[7][col] += 1
    W[4][col] += 1
    col += 1
    
    # a23 b33
    U[5][col] += 1
    V[8][col] += 1
    W[5][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
