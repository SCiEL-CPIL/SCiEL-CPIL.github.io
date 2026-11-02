# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 2, 7
    
    # We use the known rank-25 decomposition:
    # 3 blocks of 2x2*2x2 (Strassen rank 7) + 1 block of 2x2*2x1 (schoolbook rank 4)
    # Total rank = 3*7 + 4 = 25
    
    rank = 25
    
    # Initialize U, V, W as lists of lists (will be returned as such)
    # U: (n*m, rank) = (4, 25)
    # V: (m*p, rank) = (14, 25)
    # W: (p*n, rank) = (14, 25)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 * 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U factors: which A_{i,j} coefficient goes into which M_k
    # A is 2x2, indices: A00=0, A01=1, A10=2, A11=3
    strassen_U = [
        [1, 1, 1, 0, 1, -1, -1],  # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [1, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ]
    
    # V factors: which B_{i,j} coefficient goes into which M_k
    # B is 2x2, indices: B00=0, B01=1, B10=2, B11=3
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [1, 0, 0, 0, 0, 0, 1],    # B01
        [0, 0, 1, 0, 0, 1, 0],    # B10
        [1, 0, -1, 1, 1, 0, 1]    # B11
    ]
    
    # W factors: how each M_k contributes to C_{i,j}
    # C is 2x2, indices: C00=0, C01=1, C10=2, C11=3
    # C00 = M1 - M3 - M4 + M7
    # C01 = M5 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M5 - M6
    strassen_W = [
        [1, 0, -1, -1, 0, 0, 1],   # C00
        [0, 0, 0, 1, 1, 0, 0],     # C01
        [0, 1, 1, 0, 0, 0, 0],     # C10
        [1, -1, 0, 0, 1, -1, 0]    # C11
    ]
    
    # Build 3 Strassen blocks for columns (0,1), (2,3), (4,5)
    for b in range(3):
        for k in range(7):
            for i in range(4):
                U[i][col] += strassen_U[i][k]
            
            for j in range(4):
                # Map local B index j to global V index
                # Local j: 0=B00, 1=B01, 2=B10, 3=B11
                # Global V index: row * p + col
                if j == 0:  # B00
                    v_row = 0
                    v_col = 2 * b
                elif j == 1:  # B01
                    v_row = 0
                    v_col = 2 * b + 1
                elif j == 2:  # B10
                    v_row = 1
                    v_col = 2 * b
                else:  # j == 3, B11
                    v_row = 1
                    v_col = 2 * b + 1
                
                V[v_row * p + v_col][col] += strassen_V[j][k]
            
            for c_idx in range(4):
                # Map local C index to global W index
                # Local c_idx: 0=C00, 1=C01, 2=C10, 3=C11
                if c_idx == 0:  # C00
                    w_row = 2 * b
                    w_col = 0
                elif c_idx == 1:  # C01
                    w_row = 2 * b + 1
                    w_col = 0
                elif c_idx == 2:  # C10
                    w_row = 2 * b
                    w_col = 1
                else:  # c_idx == 3, C11
                    w_row = 2 * b + 1
                    w_col = 1
                
                W[w_row * n + w_col][col] += strassen_W[c_idx][k]
            
            col += 1
    
    # Schoolbook for last column (col 6)
    # C00 = A00*B06 + A01*B16
    # C10 = A10*B06 + A11*B16
    # Only 2 outputs, but we need 4 rank terms for exact decomposition
    
    # Term 1: A00 * B06 -> C00
    U[0][col] += 1
    V[0 * p + 6][col] += 1
    W[6 * n + 0][col] += 1
    col += 1
    
    # Term 2: A01 * B16 -> C00
    U[1][col] += 1
    V[1 * p + 6][col] += 1
    W[6 * n + 0][col] += 1
    col += 1
    
    # Term 3: A10 * B06 -> C10
    U[2][col] += 1
    V[0 * p + 6][col] += 1
    W[6 * n + 1][col] += 1
    col += 1
    
    # Term 4: A11 * B16 -> C10
    U[3][col] += 1
    V[1 * p + 6][col] += 1
    W[6 * n + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
