"""
Matrix multiplication 6x6x6 using Block-Strassen decomposition.
Splits 6x6 matrices into 2x2 blocks of 3x3 submatrices.
Uses Strassen's algorithm (rank 7) for the 2x2 block structure.
Uses schoolbook (rank 27) for the 3x3 submatrices.
Total rank: 7 * 27 = 189.
"""

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split the 6x6 matrix into 2x2 blocks of size 3x3.
    # The "outer" multiplication is 2x2, using Strassen (rank 7).
    # The "inner" multiplication is 3x3, using schoolbook (rank 27).
    # Total rank = 7 * 27 = 189.
    
    # Strassen coefficients for 2x2 matrix multiplication.
    # C = A * B
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the 7 Strassen terms for the 2x2 outer structure.
    # Each term is a product of a linear combination of A blocks and a linear combination of B blocks.
    # The A blocks are 3x3 matrices. The B blocks are 3x3 matrices.
    # The product of two 3x3 matrices is done via schoolbook (27 terms).
    
    # Indices for 3x3 schoolbook multiplication:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # For 3x3, there are 27 terms.
    
    # Total rank 189.
    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Helper to add a schoolbook 3x3 multiplication to the global U, V, W
    # offset_u, offset_v, offset_w are the starting indices in the 6x6 matrices for the 3x3 blocks
    # col_offset is the starting column index in U, V, W for this 3x3 multiplication
    def add_3x3_schoolbook(offset_u, offset_v, offset_w, col_offset):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # U index: (offset_u + i) * 6 + (offset_u + j) ? No.
                    # U is indexed by [i*m + j] where i is row, j is col of A.
                    # A is 6x6. Block (bi, bj) is at rows 3*bi .. 3*bi+2, cols 3*bj .. 3*bj+2.
                    # So row index in A is 3*bi + i, col index is 3*bj + j.
                    # U index = (3*bi + i) * 6 + (3*bj + j)
                    
                    u_idx = (3 * (offset_u // 3) + i) * 6 + (3 * (offset_u % 3) + j)
                    # Wait, offset_u is the block index (0..1) for row and col?
                    # Let's use block coordinates.
                    # offset_u is (bi, bj) for A.
                    # offset_v is (bj, bk) for B.
                    # offset_w is (bk, bi) for C.
                    
                    # Let's re-define the helper to take block coordinates.
                    pass

    # Let's build it directly.
    # Strassen terms:
    # Each term t in 0..6 corresponds to:
    # U_part = sum of A blocks with coefficients
    # V_part = sum of B blocks with coefficients
    # W_part = sum of C blocks with coefficients
    
    # Strassen A-block coefficients for each of the 7 terms:
    # Term 0: A00 + A11
    # Term 1: A10 + A11
    # Term 2: A00
    # Term 3: A11
    # Term 4: A00 + A01
    # Term 5: A10 - A00
    # Term 6: A01 - A11
    
    # Strassen B-block coefficients:
    # Term 0: B00 + B11
    # Term 1: B00
    # Term 2: B01 - B11
    # Term 3: B10 - B00
    # Term 4: B11
    # Term 5: B00 + B01
    # Term 6: B10 + B11
    
    # Strassen C-block coefficients:
    # Term 0: C00 (+), C11 (-), C01 (-), C10 (+) -> Wait.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Indices: C00 is (0,0), C01 is (0,1), C10 is (1,0), C11 is (1,1).
    
    strassen_a_blocks = [
        [(0,0,1), (1,1,1)], # M1
        [(1,0,1), (1,1,1)], # M2
        [(0,0,1)],          # M3
        [(1,1,1)],          # M4
        [(0,0,1), (0,1,1)], # M5
        [(1,0,1), (0,0,-1)],# M6
        [(0,1,1), (1,1,-1)] # M7
    ]
    
    strassen_b_blocks = [
        [(0,0,1), (1,1,1)], # M1
        [(0,0,1)],          # M2
        [(0,1,1), (1,1,-1)],# M3
        [(1,0,1), (0,0,-1)],# M4
        [(1,1,1)],          # M5
        [(0,0,1), (0,1,1)], # M6
        [(1,0,1), (1,1,1)]  # M7
    ]
    
    strassen_c_blocks = [
        [(0,0,1), (1,1,-1), (0,1,-1), (1,0,1)], # M1 contributes to C00, C11, C01, C10?
        # Let's check:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: C00(+1), C11(+1)
        # M2: C10(+1), C11(-1)
        # M3: C01(+1), C11(+1)
        # M4: C00(+1), C10(+1)
        # M5: C00(-1), C01(+1)
        # M6: C11(+1)
        # M7: C00(+1)
        
        [(0,0,1), (1,1,1)],       # M1
        [(1,0,1), (1,1,-1)],      # M2
        [(0,1,1), (1,1,1)],       # M3
        [(0,0,1), (1,0,1)],       # M4
        [(0,0,-1), (0,1,1)],      # M5
        [(1,1,1)],                # M6
        [(0,0,1)]                 # M7
    ]
    
    col = 0
    for t in range(7):
        a_terms = strassen_a_blocks[t]
        b_terms = strassen_b_blocks[t]
        c_terms = strassen_c_blocks[t]
        
        # For this Strassen term, we perform 27 schoolbook multiplications of 3x3 blocks.
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # U column: sum of A blocks
                    for (bi, bj, coeff) in a_terms:
                        u_row = 3 * bi + i
                        u_col = 3 * bj + j
                        U[u_row * 6 + u_col][col] += coeff
                    
                    # V column: sum of B blocks
                    for (bj, bk, coeff) in b_terms:
                        v_row = 3 * bj + k
                        v_col = 3 * bk + j
                        V[v_row * 6 + v_col][col] += coeff
                    
                    # W column: sum of C blocks
                    for (bk, bi, coeff) in c_terms:
                        w_row = 3 * bk + i
                        w_col = 3 * bi + j
                        W[w_row * 6 + w_col][col] += coeff
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
