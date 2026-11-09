"""Matrix multiplication 4x4 * 4x7 - rank-minimal tensor decomposition.

Strategy:
- For the 4x4 * 4x4 part (first 4 columns of B), use recursive Strassen giving rank 7*7 = 49.
- For the 4x4 * 4x3 part (last 3 columns of B), use schoolbook rank 4*4*3 = 48.
- Total rank = 97.

Index conventions:
- U has shape (n*m, rank) = (16, rank); U[i*m+j, r]
- V has shape (m*p, rank) = (28, rank); V[j*p+k, r]
- W has shape (p*n, rank) = (28, rank); W[k*n+i, r]

Tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
"""

DIMENSIONS = [4, 4, 7]


# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Returns a dict with keys 'U', 'V', 'W' where:
    - U is (16, rank), V is (28, rank), W is (28, rank)
    - The decomposition satisfies: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We'll build the decomposition by accumulating contributions into
    # lists of (U_col, V_col, W_col) vectors.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Part 1: 4x4 * 4x4 (first 4 columns of B) using Strassen, rank 49
    # Part 2: 4x4 * 4x3 (last 3 columns of B) using schoolbook, rank 48
    
    # === Part 1: Strassen for 4x4 * 4x4 ===
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Split B (first 4 cols) into 2x2 blocks: B = [[B00, B01], [B10, B11]]
    # Result C (first 4 cols) has 2x2 blocks: C = [[C00, C01], [C10, C11]]
    
    # Block-level Strassen: 7 intermediate products
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B01-B11)
    # M3 = A11*(B10-B00)
    # M4 = (A00+A01)*B11
    # M5 = (A10-A00)*(B00+B01)
    # M6 = (A01-A11)*(B10+B11)
    
    # Each Mk is a 2x2 matrix product, decomposed using element-level Strassen (rank 7)
    
    # For each block-level term k, define the A-block coefficients and B-block coefficients
    # A blocks: indexed by (br, bc) where br, bc in {0, 1}
    # B blocks: indexed by (br, bc) where br, bc in {0, 1}
    
    # Block-level Strassen definitions: (a_coeffs, b_coeffs)
    # a_coeffs: dict from (br,bc) to coefficient for A block
    # b_coeffs: dict from (br,bc) to coefficient for B block
    
    block_strassen = [
        # k=0: (A00+A11)*(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # k=1: (A10+A11)*B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # k=2: A00*(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # k=3: A11*(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        # k=4: (A00+A01)*B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # k=5: (A10-A00)*(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # k=6: (A01-A11)*(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
    ]
    
    # Element-level Strassen for 2x2*2x2 -> 2x2
    # For each of the 7 terms, define:
    # x_coeffs: dict from (lr,lc) to coefficient for X element (local row, col)
    # y_coeffs: dict from (lr,lc) to coefficient for Y element
    # c_coeffs: dict from (lr,lc) to coefficient for C element (output)
    
    elem_strassen = [
        # t=0: (X00+X11)*(Y00+Y11) -> C00+=1, C11-=1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}),
        # t=1: (X10+X11)*Y00 -> C10+=1, C11-=1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # t=2: X00*(Y01-Y11) -> C01+=1, C11+=1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # t=3: X11*(Y10-Y00) -> C00-=1, C10+=1
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): -1, (1,0): 1}),
        # t=4: (X00+X01)*Y11 -> C01-=1, C11+=1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): -1, (1,1): 1}),
        # t=5: (X10-X00)*(Y00+Y01) -> C00+=1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1}),
        # t=6: (X01-X11)*(Y10+Y11) -> C00-=1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1}),
    ]
    
    # For each block-level term k and element-level term t, we get a rank-1 component
    # The U vector has non-zero entries at positions corresponding to A elements
    # The V vector has non-zero entries at positions corresponding to B elements
    # The W vector has non-zero entries at positions corresponding to C elements
    
    for k in range(7):
        a_block_coeffs, b_block_coeffs = block_strassen[k]
        for t in range(7):
            x_coeffs, y_coeffs, c_coeffs = elem_strassen[t]
            
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            
            # For each A block in a_block_coeffs and each element in x_coeffs:
            for (abr, abc), a_coeff in a_block_coeffs.items():
                for (xl, xc), x_coeff in x_coeffs.items():
                    # Global A index: row = abr*2 + xl, col = abc*2 + xc
                    gi = abr * 2 + xl
                    gj = abc * 2 + xc
                    U_col[gi * m + gj] = a_coeff * x_coeff
            
            # For each B block in b_block_coeffs and each element in y_coeffs:
            for (bbr, bbc), b_coeff in b_block_coeffs.items():
                for (yl, yc), y_coeff in y_coeffs.items():
                    # Global B index: row = bbr*2 + yl, col = bbc*2 + yc
                    gj = bbr * 2 + yl
                    gk = bbc * 2 + yc
                    V_col[gj * p + gk] = b_coeff * y_coeff
            
            # For each C element in c_coeffs:
            # C block position: the output block is at (cr, cc) where
            # cr is determined by which row of the output this contributes to
            # cc is determined by which column
            # Actually, the Strassen formula gives:
            # C00 = M0 - M3 + M5 - M6
            # C01 = M2 + M4
            # C10 = M1 + M3
            # C11 = M0 - M1 + M4 + M6
            # So we need to know which output block each M_k contributes to and with what sign
            
            # Let me define the output block contributions for each M_k:
            # M0: C00(+1), C11(-1)
            # M1: C10(+1), C11(-1)
            # M2: C01(+1), C11(+1)
            # M3: C00(-1), C10(+1)
            # M4: C01(-1), C11(+1)
            # M5: C00(+1)
            # M6: C00(-1), C11(-1)
            
            output_blocks = {
                0: [(0, 0, 1), (1, 1, -1)],
                1: [(1, 0, 1), (1, 1, -1)],
                2: [(0, 1, 1), (1, 1, 1)],
                3: [(0, 0, -1), (1, 0, 1)],
                4: [(0, 1, -1), (1, 1, 1)],
                5: [(0, 0, 1)],
                6: [(0, 0, -1), (1, 1, -1)],
            }
            
            for (ocr, occ, o_coeff) in output_blocks[k]:
                for (cl, cc), c_coeff in c_coeffs.items():
                    # Global C index: row = ocr*2 + cl, col = occ*2 + cc
                    gi = ocr * 2 + cl
                    gk = occ * 2 + cc
                    W_col[gk * n + gi] = o_coeff * c_coeff
            
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
    
    # === Part 2: Schoolbook for 4x4 * 4x3 (last 3 columns of B) ===
    # For each (i, j, k) where k is in {4, 5, 6}:
    for i in range(n):
        for j in range(m):
            for k in range(4, p):  # k = 4, 5, 6
                U_col = [0.0] * (n * m)
                V_col = [0.0] * (m * p)
                W_col = [0.0] * (p * n)
                
                U_col[i * m + j] = 1.0
                V_col[j * p + k] = 1.0
                W_col[k * n + i] = 1.0
                
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)
    
    rank = len(U_cols)
    
    # Build the final matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for idx, val in enumerate(U_cols[r]):
            if val != 0.0:
                U[idx][r] = val
        for idx, val in enumerate(V_cols[r]):
            if val != 0.0:
                V[idx][r] = val
        for idx, val in enumerate(W_cols[r]):
            if val != 0.0:
                W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
