import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Rank calculation:
    # 4 blocks of 2x2x2 Strassen (7 each) = 28
    # 4 rank-1 terms for last row = 4
    # Total = 32
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen_block(i_off, j_off, k_off, col_start):
        # i_off: 0 or 1 (offset for i in 0..1)
        # j_off: 0, 1, 2, 3 (offset for j)
        # k_off: 0, 1, 2, 3 (offset for k)
        
        # Strassen coefficients for 2x2x2
        # P1: (A00+A11)(B00+B11)
        # P2: (A10+A11)B00
        # P3: A00(B01-B11)
        # P4: A11(B10-B00)
        # P5: (A00+A01)B11
        # P6: (A10-A00)(B00+B01)
        # P7: (A01-A11)(B10+B11)
        
        # We define the U, V, W contributions for each P
        
        # P1
        c = col_start
        # U: A00, A11
        U[(i_off+0)*m + (j_off+0)][c] = 1
        U[(i_off+1)*m + (j_off+1)][c] = 1
        # V: B00, B10, B01, B11 (all 4)
        V[(j_off+0)*p + (k_off+0)][c] = 1
        V[(j_off+1)*p + (k_off+0)][c] = 1
        V[(j_off+0)*p + (k_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        # W: C00, C10, C01, C11 (all 4)
        W[(k_off+0)*n + (i_off+0)][c] = 1
        W[(k_off+1)*n + (i_off+0)][c] = 1
        W[(k_off+0)*n + (i_off+1)][c] = 1
        W[(k_off+1)*n + (i_off+1)][c] = 1
        
        # P2
        c = col_start + 1
        # U: A10, A11
        U[(i_off+1)*m + (j_off+0)][c] = 1
        U[(i_off+1)*m + (j_off+1)][c] = 1
        # V: B00
        V[(j_off+0)*p + (k_off+0)][c] = 1
        # W: C10
        W[(k_off+0)*n + (i_off+1)][c] = 1
        
        # P3
        c = col_start + 2
        # U: A00
        U[(i_off+0)*m + (j_off+0)][c] = 1
        # V: B01, -B11
        V[(j_off+0)*p + (k_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = -1
        # W: C01
        W[(k_off+1)*n + (i_off+0)][c] = 1
        
        # P4
        c = col_start + 3
        # U: A11
        U[(i_off+1)*m + (j_off+1)][c] = 1
        # V: B10, -B00
        V[(j_off+1)*p + (k_off+0)][c] = 1
        V[(j_off+0)*p + (k_off+0)][c] = -1
        # W: C10
        W[(k_off+0)*n + (i_off+1)][c] = 1
        
        # P5
        c = col_start + 4
        # U: A00, A01
        U[(i_off+0)*m + (j_off+0)][c] = 1
        U[(i_off+0)*m + (j_off+1)][c] = 1
        # V: B11
        V[(j_off+1)*p + (k_off+1)][c] = 1
        # W: C00
        W[(k_off+0)*n + (i_off+0)][c] = 1
        
        # P6
        c = col_start + 5
        # U: A10, -A00
        U[(i_off+1)*m + (j_off+0)][c] = 1
        U[(i_off+0)*m + (j_off+0)][c] = -1
        # V: B00, B01
        V[(j_off+0)*p + (k_off+0)][c] = 1
        V[(j_off+0)*p + (k_off+1)][c] = 1
        # W: C11
        W[(k_off+1)*n + (i_off+1)][c] = 1
        
        # P7
        c = col_start + 6
        # U: A01, -A11
        U[(i_off+0)*m + (j_off+1)][c] = 1
        U[(i_off+1)*m + (j_off+1)][c] = -1
        # V: B10, B11
        V[(j_off+1)*p + (k_off+0)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        # W: C00, C01?
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        # P7 contributes to C00.
        W[(k_off+0)*n + (i_off+0)][c] = 1

    # Block 1: i=0..1, j=0..1, k=0..1
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: i=0..1, j=2..3, k=0..1
    add_strassen_block(0, 2, 0, 7)
    
    # Block 3: i=0..1, j=0..1, k=2..3
    add_strassen_block(0, 0, 2, 14)
    
    # Block 4: i=0..1, j=2..3, k=2..3
    add_strassen_block(0, 2, 2, 21)
    
    # Last row terms: i=2
    # j=0
    c = 28
    for k in range(p):
        U[2*m + 0][c] = 1
        V[0*p + k][c] = 1
        W[k*n + 2][c] = 1
        
    # j=1
    c = 29
    for k in range(p):
        U[2*m + 1][c] = 1
        V[1*p + k][c] = 1
        W[k*n + 2][c] = 1
        
    # j=2
    c = 30
    for k in range(p):
        U[2*m + 2][c] = 1
        V[2*p + k][c] = 1
        W[k*n + 2][c] = 1
        
    # j=3
    c = 31
    for k in range(p):
        U[2*m + 3][c] = 1
        V[3*p + k][c] = 1
        W[k*n + 2][c] = 1

    return {"U": U, "V": V, "W": W}
