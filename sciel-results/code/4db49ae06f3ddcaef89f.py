import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms definitions
    # Each term is a tuple (coeffs_A, coeffs_B, coeffs_C)
    # coeffs_A: [c00, c01, c10, c11] for A00, A01, A10, A11
    # coeffs_B: [c00, c01, c10, c11] for B00, B01, B10, B11
    # coeffs_C: [c00, c01, c10, c11] for C00, C01, C10, C11 (coefficients for reconstruction)
    
    terms = [
        # M1: (A00 + A11)(B00 + B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]), # C00+=1, C11-=1
        # M2: (A10 + A11)B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # C10+=1
        # M3: A00(B01 - B11)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]), # C01+=1, C11-=1
        # M4: A11(B10 - B00)
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]), # C00-=1, C10+=1
        # M5: (A00 + A01)B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]), # C01+=1, C11+=1
        # M6: (A10 - A00)(B00 + B01)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [-1, 0, 0, 0]), # C00-=1
        # M7: (A01 - A11)(B10 + B11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]), # C00+=1, C11-=1
    ]
    
    for g in range(7):
        cA = terms[g][0]
        cB = terms[g][1]
        cC = terms[g][2]
        
        # cC corresponds to C00, C01, C10, C11
        # C00 is block (0,0) -> C cols 0-3, row 0
        # C01 is block (0,1) -> C cols 4-7, row 0
        # C10 is block (1,0) -> C cols 0-3, row 1
        # C11 is block (1,1) -> C cols 4-7, row 1
        
        # For each j (0,1) and k (0..3)
        for j in range(2):
            for k in range(4):
                # U part: s_g_j
                # s_g_j is linear combo of A entries
                # A00: row 0, cols 0-1. Entry j is at (0, j)
                # A01: row 0, cols 2-3. Entry j is at (0, 2+j)
                # A10: row 1, cols 0-1. Entry j is at (1, j)
                # A11: row 1, cols 2-3. Entry j is at (1, 2+j)
                
                U[0*4 + j][col] += cA[0]
                U[0*4 + 2 + j][col] += cA[1]
                U[1*4 + j][col] += cA[2]
                U[1*4 + 2 + j][col] += cA[3]
                
                # V part: t_g_j_k
                # B00: rows 0-1, cols 0-3. Entry (j,k) is at (j, k)
                # B01: rows 0-1, cols 4-7. Entry (j,k) is at (j, 4+k)
                # B10: rows 2-3, cols 0-3. Entry (j,k) is at (2+j, k)
                # B11: rows 2-3, cols 4-7. Entry (j,k) is at (2+j, 4+k)
                
                V[j*8 + k][col] += cB[0]
                V[j*8 + 4 + k][col] += cB[1]
                V[(2+j)*8 + k][col] += cB[2]
                V[(2+j)*8 + 4 + k][col] += cB[3]
                
                # W part:
                # The term contributes to P_g[k]
                # P_g[k] is added to C blocks with coefficients cC
                # C00 (block 0,0): coeff cC[0]. Col in C: 0*4 + k = k. Row: 0.
                # C01 (block 0,1): coeff cC[1]. Col in C: 1*4 + k = 4+k. Row: 0.
                # C10 (block 1,0): coeff cC[2]. Col in C: 0*4 + k = k. Row: 1.
                # C11 (block 1,1): coeff cC[3]. Col in C: 1*4 + k = 4+k. Row: 1.
                
                # W index: col_C * n + row_C
                if cC[0] != 0:
                    W[k*2 + 0][col] += cC[0]
                if cC[1] != 0:
                    W[(4+k)*2 + 0][col] += cC[1]
                if cC[2] != 0:
                    W[k*2 + 1][col] += cC[2]
                if cC[3] != 0:
                    W[(4+k)*2 + 1][col] += cC[3]
                    
                col += 1
                
    return {"U": U, "V": V, "W": W}
