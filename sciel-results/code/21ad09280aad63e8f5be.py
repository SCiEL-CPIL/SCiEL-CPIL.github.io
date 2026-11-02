import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    dim_A = n * m   # 8
    dim_B = m * p   # 24
    dim_C = p * n   # 12
    
    rank = 42
    
    U = [[0] * rank for _ in range(dim_A)]
    V = [[0] * rank for _ in range(dim_B)]
    W = [[0] * rank for _ in range(dim_C)]
    
    # Standard Strassen 2x2x2 rank-7 decomposition
    # For X * Y = Z where X, Y, Z are 2x2 matrices
    # X indexed as X[i,j], Y indexed as Y[j,k], Z indexed as Z[i,k]
    #
    # 7 products:
    # P1 = (X00+X11)(Y00+Y11)
    # P2 = (X10+X11)Y00
    # P3 = X00(Y01-Y11)
    # P4 = X11(Y10-Y00)
    # P5 = (X00+X01)Y11
    # P6 = (X10-X00)(Y00+Y01)
    # P7 = (X01-X11)(Y10+Y11)
    #
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # U_local[t][i*2+j] = coefficient of X[i,j] in product t
    U_local = [
        [1, 0, 0, 1],   # P1: X00 + X11
        [0, 0, 1, 1],   # P2: X10 + X11
        [1, 0, 0, 0],   # P3: X00
        [0, 0, 0, 1],   # P4: X11
        [1, 1, 0, 0],   # P5: X00 + X01
        [-1, 0, 1, 0],  # P6: X10 - X00
        [0, 1, 0, -1]   # P7: X01 - X11
    ]
    
    # V_local[t][j*2+k] = coefficient of Y[j,k] in product t
    V_local = [
        [1, 0, 0, 1],   # P1: Y00 + Y11
        [1, 0, 0, 0],   # P2: Y00
        [0, 1, 0, -1],  # P3: Y01 - Y11
        [-1, 0, 1, 0],  # P4: Y10 - Y00
        [0, 0, 0, 1],   # P5: Y11
        [1, 1, 0, 0],   # P6: Y00 + Y01
        [0, 0, 1, 1]    # P7: Y10 + Y11
    ]
    
    # W_local[t][i*2+k] = coefficient of Z[i,k] in product t
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    W_local = [
        [1, 0, 0, 1],   # P1: Z00 + Z11
        [0, 0, 1, -1],  # P2: Z10 - Z11
        [0, 1, 0, 1],   # P3: Z01 + Z11
        [1, 0, 1, 0],   # P4: Z00 + Z10
        [-1, 1, 0, 0],  # P5: -Z00 + Z01
        [0, 0, 0, 1],   # P6: Z11
        [1, 0, 0, 0]    # P7: Z00
    ]
    
    col_offset = 0
    
    for cb in range(3):
        for side in ['L', 'R']:
            for t in range(7):
                r_col = col_offset
                
                # Determine which 2x2 block of A and B to use
                if side == 'L':
                    a_col_base = 0      # A columns 0,1
                    b_row_base = 0      # B rows 0,1
                else:
                    a_col_base = 2      # A columns 2,3
                    b_row_base = 2      # B rows 2,3
                
                # C column block: columns 2*cb, 2*cb+1
                c_col_base = 2 * cb
                
                # Fill U: A[i, a_col_base + j] where i in {0,1}, j in {0,1}
                # U row index = i * m + (a_col_base + j)
                for i in range(2):
                    for j in range(2):
                        val = U_local[t][i * 2 + j]
                        if val != 0:
                            ga = i * m + (a_col_base + j)
                            U[ga][r_col] += val
                
                # Fill V: B[b_row_base + j, c_col_base + k] where j in {0,1}, k in {0,1}
                # V row index = (b_row_base + j) * p + (c_col_base + k)
                for j in range(2):
                    for k in range(2):
                        val = V_local[t][j * 2 + k]
                        if val != 0:
                            gb = (b_row_base + j) * p + (c_col_base + k)
                            V[gb][r_col] += val
                
                # Fill W: C[i, c_col_base + k] where i in {0,1}, k in {0,1}
                # W row index = (c_col_base + k) * n + i  (trace convention: W[k*n+i, r])
                for i in range(2):
                    for k in range(2):
                        val = W_local[t][i * 2 + k]
                        if val != 0:
                            gc = (c_col_base + k) * n + i
                            W[gc][r_col] += val
                
                col_offset += 1
    
    return {"U": U, "V": "V" if False else V, "W": W}
