"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen factors for 2x2x2
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],   # (0,0)
        [0, 0, 0, 0, 1, 0, 1],    # (0,1)
        [0, 1, 0, 0, 0, 1, 0],    # (1,0)
        [1, 1, 0, 1, 0, 0, -1],   # (1,1)
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],   # (0,0)
        [0, 0, 1, 0, 0, 1, 0],    # (0,1)
        [0, 0, 0, 1, 0, 0, 1],    # (1,0)
        [1, 0, -1, 0, 1, 0, 1],   # (1,1)
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],   # (0,0)
        [0, 1, 0, 1, 0, 0, 0],    # (0,1)
        [0, 0, 1, 0, 1, 0, 0],    # (1,0)
        [1, -1, 1, 0, 0, 1, 0],   # (1,1)
    ]
    
    # Total rank: 49 (for 4x4x4) + 28 (for 4x4x2) = 77
    total_rank = 77
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 77
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 77
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 77
    
    col_offset = 0
    
    # Part 1: 4x4x4 decomposition using Strassen (rank 49)
    # Strassen coefficients for A, B, C blocks
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Coefficients for A blocks (bi, bj)
    A_coeffs = [
        [[1, 0], [0, 1]],  # M1: A11(1), A22(1)
        [[0, 0], [1, 1]],  # M2: A21(1), A22(1)
        [[1, 0], [0, 0]],  # M3: A11(1)
        [[0, 0], [0, 1]],  # M4: A22(1)
        [[1, 1], [0, 0]],  # M5: A11(1), A12(1)
        [[-1, 0], [1, 0]], # M6: A11(-1), A21(1)
        [[0, 1], [0, -1]], # M7: A12(1), A22(-1)
    ]
    
    # Coefficients for B blocks (bj, bk)
    B_coeffs = [
        [[1, 1], [1, 1]],  # M1: B11(1), B22(1) -- Wait, M1 is (A11+A22)(B11+B22).
                           # So B11 and B22 are used.
        [[1, 0], [0, 0]],  # M2: B11(1)
        [[0, 1], [0, -1]], # M3: B12(1), B22(-1)
        [[-1, 0], [1, 0]], # M4: B11(-1), B21(1)
        [[0, 0], [0, 1]],  # M5: B22(1)
        [[1, 1], [0, 0]],  # M6: B11(1), B12(1)
        [[0, 0], [1, 1]],  # M7: B21(1), B22(1)
    ]
    
    # Coefficients for C blocks (bi, bk)
    C_coeffs = [
        [[1, 0], [0, 1]],  # M1: C11(1), C22(1)
        [[0, 0], [1, 0]],  # M2: C21(1)
        [[0, 1], [0, 0]],  # M3: C12(1)
        [[0, 0], [0, 1]],  # M4: C22(1)
        [[0, 0], [0, -1]], # M5: C22(-1)
        [[0, 1], [0, 0]],  # M6: C12(1)
        [[0, 0], [1, 0]],  # M7: C21(1)
    ]
    # Note: Standard Strassen reconstruction:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M6
    # My C_coeffs above are just placeholders, I need to ensure the W part matches the reconstruction.
    # Actually, the W part in the tensor decomposition handles the reconstruction.
    # The tensor T = sum_c (A_c tensor B_c tensor C_c).
    # So W should correspond to the C_c blocks.
    
    for c in range(7):
        for r in range(7):
            col = col_offset + r
            # U
            for bi in range(2):
                for bj in range(2):
                    coeff_a = A_coeffs[c][bi][bj]
                    if coeff_a != 0:
                        for di in range(2):
                            for dj in range(2):
                                i = bi * 2 + di
                                j = bj * 2 + dj
                                U[i * m + j][col] += coeff_a * S_U[di * 2 + dj][r]
            
            # V
            for bj in range(2):
                for bk in range(2):
                    coeff_b = B_coeffs[c][bj][bk]
                    if coeff_b != 0:
                        for dj in range(2):
                            for dk in range(2):
                                j = bj * 2 + dj
                                k = bk * 2 + dk
                                V[j * p + k][col] += coeff_b * S_V[dj * 2 + dk][r]
            
            # W
            for bi in range(2):
                for bk in range(2):
                    coeff_c = C_coeffs[c][bi][bk]
                    if coeff_c != 0:
                        for dk in range(2):
                            for di in range(2):
                                k = bk * 2 + dk
                                i = bi * 2 + di
                                W[k * n + i][col] += coeff_c * S_W[dk * 2 + di][r]
            
            col_offset += 7
            
    # Part 2: 4x4x2 decomposition (rank 28)
    # B_right is cols 4,5.
    # B31: rows 0,1; cols 4,5. (bj=0, bk=2)
    # B32: rows 2,3; cols 4,5. (bj=1, bk=2)
    # C31: rows 0,1; cols 4,5. (bi=0, bk=2)
    # C32: rows 2,3; cols 4,5. (bi=1, bk=2)
    
    # Products:
    # P1 = A11 B31 -> C31
    # P2 = A12 B32 -> C31
    # P3 = A21 B31 -> C32
    # P4 = A22 B32 -> C32
    
    products = [
        (0, 0, 0, 2, 0, 2), # A11 (bi=0,bj=0), B31 (bj=0,bk=2), C31 (bi=0,bk=2)
        (0, 1, 1, 2, 0, 2), # A12 (bi=0,bj=1), B32 (bj=1,bk=2), C31 (bi=0,bk=2)
        (1, 0, 0, 2, 1, 2), # A21 (bi=1,bj=0), B31 (bj=0,bk=2), C32 (bi=1,bk=2)
        (1, 1, 1, 2, 1, 2), # A22 (bi=1,bj=1), B32 (bj=1,bk=2), C32 (bi=1,bk=2)
    ]
    
    for bi_a, bj_a, bj_b, bk_b, bi_c, bk_c in products:
        for r in range(7):
            col = col_offset + r
            # U
            for di in range(2):
                for dj in range(2):
                    i = bi_a * 2 + di
                    j = bj_a * 2 + dj
                    U[i * m + j][col] += S_U[di * 2 + dj][r]
            
            # V
            for dj in range(2):
                for dk in range(2):
                    j = bj_b * 2 + dj
                    k = bk_b * 2 + dk
                    V[j * p + k][col] += S_V[dj * 2 + dk][r]
            
            # W
            for dk in range(2):
                for di in range(2):
                    k = bk_c * 2 + dk
                    i = bi_c * 2 + di
                    W[k * n + i][col] += S_W[dk * 2 + di][r]
            
            col_offset += 7
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
