import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen 2x2x2 decomposition
    # A is 2x2, B is 2x2, C is 2x2
    # U indices: (i, j) -> i*2 + j for A
    # V indices: (j, k) -> j*2 + k for B
    # W indices: (k, i) -> k*2 + i for C (Trace convention W[k*n+i, r] -> W[k*2+i, r])
    # Wait, W is p*n. Here local p=2, n=2.
    # Global W index: k_global * n + i_global.
    
    def add_strassen(offset_u, offset_v, offset_w, start_col):
        # Strassen factors
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b21-b22)
        # M4 = a22(b12-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # We need to map these to U, V, W
        # U[i*m+j, col], V[j*p+k, col], W[k*n+i, col]
        
        # Local indices:
        # A: 0,1,2,3 (0:11, 1:12, 2:21, 3:22)
        # B: 0,1,2,3 (0:11, 1:12, 2:21, 3:22)
        # C: 0,1,2,3 (0:11, 1:12, 2:21, 3:22)
        
        # M1
        col = start_col
        # U: a11 (0), a22 (3) -> 1
        U[offset_u + 0][col] = 1
        U[offset_u + 3][col] = 1
        # V: b11 (0), b22 (3) -> 1
        V[offset_v + 0][col] = 1
        V[offset_v + 3][col] = 1
        # W: c11 (0) += 1, c22 (3) += 1, c12 (1) -= 1, c21 (2) -= 1
        W[offset_w + 0][col] = 1
        W[offset_w + 3][col] = 1
        W[offset_w + 1][col] = -1
        W[offset_w + 2][col] = -1
        col += 1
        
        # M2
        # U: a21 (2), a22 (3) -> 1
        U[offset_u + 2][col] = 1
        U[offset_u + 3][col] = 1
        # V: b11 (0) -> 1
        V[offset_v + 0][col] = 1
        # W: c21 (2) += 1, c22 (3) += 1
        W[offset_w + 2][col] = 1
        W[offset_w + 3][col] = 1
        col += 1
        
        # M3
        # U: a11 (0) -> 1
        U[offset_u + 0][col] = 1
        # V: b21 (2), b22 (3) -> 1, -1
        V[offset_v + 2][col] = 1
        V[offset_v + 3][col] = -1
        # W: c11 (0) += 1, c12 (1) += 1
        W[offset_w + 0][col] = 1
        W[offset_w + 1][col] = 1
        col += 1
        
        # M4
        # U: a22 (3) -> 1
        U[offset_u + 3][col] = 1
        # V: b12 (1), b11 (0) -> 1, -1
        V[offset_v + 1][col] = 1
        V[offset_v + 0][col] = -1
        # W: c21 (2) += 1, c22 (3) += 1
        W[offset_w + 2][col] = 1
        W[offset_w + 3][col] = 1
        col += 1
        
        # M5
        # U: a11 (0), a12 (1) -> 1
        U[offset_u + 0][col] = 1
        U[offset_u + 1][col] = 1
        # V: b22 (3) -> 1
        V[offset_v + 3][col] = 1
        # W: c11 (0) += 1, c22 (3) += 1
        W[offset_w + 0][col] = 1
        W[offset_w + 3][col] = 1
        col += 1
        
        # M6
        # U: a21 (2), a11 (0) -> 1, -1
        U[offset_u + 2][col] = 1
        U[offset_u + 0][col] = -1
        # V: b11 (0), b12 (1) -> 1
        V[offset_v + 0][col] = 1
        V[offset_v + 1][col] = 1
        # W: c21 (2) += 1, c22 (3) += 1
        W[offset_w + 2][col] = 1
        W[offset_w + 3][col] = 1
        col += 1
        
        # M7
        # U: a12 (1), a22 (3) -> 1, -1
        U[offset_u + 1][col] = 1
        U[offset_u + 3][col] = -1
        # V: b21 (2), b22 (3) -> 1
        V[offset_v + 2][col] = 1
        V[offset_v + 3][col] = 1
        # W: c11 (0) += 1, c12 (1) += 1
        W[offset_w + 0][col] = 1
        W[offset_w + 1][col] = 1
        col += 1
        
        return col

    # 1. A B1 (2x4 by 4x4)
    # A_L is rows 0-1, cols 0-1 of A. Offset U: 0.
    # A_R is rows 0-1, cols 2-3 of A. Offset U: 2.
    # B1 is rows 0-3, cols 0-3 of B.
    # B11: rows 0-1, cols 0-1. Offset V: 0.
    # B12: rows 0-1, cols 2-3. Offset V: 2.
    # B21: rows 2-3, cols 0-1. Offset V: 4.
    # B22: rows 2-3, cols 2-3. Offset V: 6.
    
    # C1 is cols 0-1 of C. Offset W: 0.
    # C2 is cols 2-3 of C. Offset W: 2.
    
    col = 0
    # P1 = A_L B11
    col = add_strassen(0, 0, 0, col)
    # P2 = A_R B21
    col = add_strassen(2, 4, 0, col)
    # P3 = A_L B12
    col = add_strassen(0, 2, 2, col)
    # P4 = A_R B22
    col = add_strassen(2, 6, 2, col)
    
    # 2. A B2 (2x4 by 4x2)
    # B2 is rows 0-3, cols 4-5 of B.
    # B5: rows 0-1, cols 4-5. Offset V: 8.
    # B6: rows 2-3, cols 4-5. Offset V: 10.
    # C3 is cols 4-5 of C. Offset W: 4.
    
    # P5 = A_L B5
    col = add_strassen(0, 8, 4, col)
    # P6 = A_R B6
    col = add_strassen(2, 10, 4, col)
    
    return {"U": U, "V": V, "W": W}
