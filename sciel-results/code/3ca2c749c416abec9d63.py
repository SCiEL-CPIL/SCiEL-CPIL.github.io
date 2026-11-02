import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen 2x2 decomposition
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = [[z00, z01], [z10, z11]]
    
    # M1 = (x00 + x11)(y00 + y11)
    # M2 = (x10 + x11)y00
    # M3 = x00(y01 - y11)
    # M4 = x11(y10 - y00)
    # M5 = (x00 + x01)y11
    # M6 = (x10 - x00)(y00 + y01)
    # M7 = (x01 - x11)(y10 + y11)
    
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6

    def add_strassen_block(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, col_offset):
        # A is 2x2 block at a_row_start, a_col_start
        # B is 2x2 block at b_row_start, b_col_start
        # C is 2x2 block at c_row_start, c_col_start
        
        # A indices
        a00 = a_row_start * m + a_col_start
        a01 = a_row_start * m + a_col_start + 1
        a10 = (a_row_start + 1) * m + a_col_start
        a11 = (a_row_start + 1) * m + a_col_start + 1
        
        # B indices
        b00 = b_row_start * p + b_col_start
        b01 = b_row_start * p + b_col_start + 1
        b10 = (b_row_start + 1) * p + b_col_start
        b11 = (b_row_start + 1) * p + b_col_start + 1
        
        # C indices (W index: k*n + i)
        # C is 2x2 at c_row_start (i), c_col_start (k)
        c00 = c_col_start * n + c_row_start
        c01 = (c_col_start + 1) * n + c_row_start
        c10 = c_col_start * n + (c_row_start + 1)
        c11 = (c_col_start + 1) * n + (c_row_start + 1)
        
        # 7 Multiplications
        # col 0: M1
        U[a00][col_offset] += 1
        U[a11][col_offset] += 1
        V[b00][col_offset] += 1
        V[b11][col_offset] += 1
        W[c00][col_offset] += 1
        W[c11][col_offset] += 1
        
        # col 1: M2
        U[a10][col_offset+1] += 1
        U[a11][col_offset+1] += 1
        V[b00][col_offset+1] += 1
        W[c10][col_offset+1] += 1
        W[c11][col_offset+1] -= 1
        
        # col 2: M3
        U[a00][col_offset+2] += 1
        V[b01][col_offset+2] += 1
        V[b11][col_offset+2] -= 1
        W[c01][col_offset+2] += 1
        W[c11][col_offset+2] += 1
        
        # col 3: M4
        U[a11][col_offset+3] += 1
        V[b10][col_offset+3] += 1
        V[b00][col_offset+3] -= 1
        W[c00][col_offset+3] += 1
        W[c10][col_offset+3] += 1
        
        # col 4: M5
        U[a00][col_offset+4] += 1
        U[a01][col_offset+4] += 1
        V[b11][col_offset+4] += 1
        W[c00][col_offset+4] -= 1
        W[c01][col_offset+4] += 1
        
        # col 5: M6
        U[a10][col_offset+5] += 1
        U[a00][col_offset+5] -= 1
        V[b00][col_offset+5] += 1
        V[b01][col_offset+5] += 1
        W[c11][col_offset+5] += 1
        
        # col 6: M7
        U[a01][col_offset+6] += 1
        U[a11][col_offset+6] -= 1
        V[b10][col_offset+6] += 1
        V[b11][col_offset+6] += 1
        W[c00][col_offset+6] += 1

    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 2, 2, 0, 0, 0, 7)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 0, 0, 2, 0, 2, 14)
    
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 2, 2, 2, 0, 2, 21)
    
    # Block 4: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen_block(0, 0, 0, 4, 0, 4, 28)
    
    # Block 5: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    add_strassen_block(0, 2, 2, 4, 0, 4, 35)

    return {"U": U, "V": V, "W": W}
