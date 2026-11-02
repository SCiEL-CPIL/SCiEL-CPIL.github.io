import time

def construct():
    n, m, p = 2, 2, 6
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B01
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M5
    # C10 = M3 + M6
    # C11 = M1 - M4 + M2 - M3
    
    # We have 3 blocks of B (cols 0-1, 2-3, 4-5)
    # For each block s (0, 1, 2), we compute 7 products.
    # Total 21 products.
    
    for s in range(3):
        col_offset = 2 * s
        for m_idx in range(7):
            r = s * 7 + m_idx
            
            # Define U, V, W for this product
            # U is 4x1, V is 12x1, W is 12x1
            
            if m_idx == 0:
                # M1 = (A00+A11)(B00+B11)
                U[0][r] = 1 # A00
                U[3][r] = 1 # A11
                V[0 * p + col_offset][r] = 1 # B0, col0
                V[1 * p + col_offset + 1][r] = 1 # B1, col1
                
                # C00 += M1
                W[col_offset * n + 0][r] = 1
                # C11 += M1
                W[(col_offset + 1) * n + 1][r] = 1
                
            elif m_idx == 1:
                # M2 = (A10+A11)B01
                U[2][r] = 1 # A10
                U[3][r] = 1 # A11
                V[0 * p + col_offset + 1][r] = 1 # B0, col1
                
                # C01 += M2
                W[(col_offset + 1) * n + 0][r] = 1
                # C11 += M2
                W[(col_offset + 1) * n + 1][r] = 1
                
            elif m_idx == 2:
                # M3 = A00(B10-B11)
                U[0][r] = 1 # A00
                V[1 * p + col_offset][r] = 1 # B1, col0
                V[1 * p + col_offset + 1][r] = -1 # B1, col1
                
                # C10 += M3
                W[col_offset * n + 1][r] = 1
                # C11 -= M3
                W[(col_offset + 1) * n + 1][r] = -1
                
            elif m_idx == 3:
                # M4 = A11(B00-B10)
                U[3][r] = 1 # A11
                V[0 * p + col_offset][r] = 1 # B0, col0
                V[1 * p + col_offset][r] = -1 # B1, col0
                
                # C00 += M4
                W[col_offset * n + 0][r] = 1
                # C11 -= M4
                W[(col_offset + 1) * n + 1][r] = -1
                
            elif m_idx == 4:
                # M5 = (A00+A01)B11
                U[0][r] = 1 # A00
                U[1][r] = 1 # A01
                V[1 * p + col_offset + 1][r] = 1 # B1, col1
                
                # C00 -= M5
                W[col_offset * n + 0][r] = -1
                # C01 += M5
                W[(col_offset + 1) * n + 0][r] = 1
                
            elif m_idx == 5:
                # M6 = (A10-A00)(B00+B01)
                U[2][r] = 1 # A10
                U[0][r] = -1 # A00
                V[0 * p + col_offset][r] = 1 # B0, col0
                V[0 * p + col_offset + 1][r] = 1 # B0, col1
                
                # C10 += M6
                W[col_offset * n + 1][r] = 1
                
            elif m_idx == 6:
                # M7 = (A01-A11)(B10+B11)
                U[1][r] = 1 # A01
                U[3][r] = -1 # A11
                V[1 * p + col_offset][r] = 1 # B1, col0
                V[1 * p + col_offset + 1][r] = 1 # B1, col1
                
                # C00 += M7
                W[col_offset * n + 0][r] = 1
                # C10 += M7 ? No, C10 = M3 + M6.
                # C11 += M7
                W[(col_offset + 1) * n + 1][r] = 1

    return {"U": U, "V": V, "W": W}
