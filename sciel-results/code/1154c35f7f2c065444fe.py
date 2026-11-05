import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    DEADLINE = 130

    # Strategy: 4x4x5 = 4x4x4 + 4x4x1
    # 4x4x4: Winograd F(4) rank 40
    # 4x4x1: schoolbook rank 16
    # Total: 56

    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # 16 x 56
    V = [[0] * rank for _ in range(m * p)]   # 20 x 56
    W = [[0] * rank for _ in range(p * n)]   # 20 x 56

    # Winograd F(4) for 4x4x4
    # The F(4) transform uses 16 intermediate values from A (4x4) and 16 from B (4x4)
    # producing 40 scalar multiplications
    #
    # F(4) basis vectors for rows of A (16 vectors in R^4):
    fA = [
        [1, 0, 0, 0],   # 0: A[0,0]
        [0, 0, 0, 1],   # 1: A[0,3]
        [1, 0, 0, 1],   # 2: A[0,0]+A[0,3]
        [0, 1, -1, 0],  # 3: A[0,1]-A[0,2]
        [0, 1, 1, 0],   # 4: A[0,1]+A[0,2]
        [0, 1, 1, 0],   # 5
        [0, 1, 1, 0],   # 6
        [0, 1, 0, 0],   # 7: A[0,1]
        [0, 1, 0, 0],   # 8
        [0, 1, 0, 0],   # 9
        [0, 0, 1, 0],   # 10: A[0,2]
        [0, 0, 1, 0],   # 11
        [0, 0, 1, 0],   # 12
        [1, 0, 0, -1],  # 13: A[0,0]-A[0,3]
        [1, 0, 0, -1],  # 14
        [1, 0, 0, -1],  # 15
    ]
    # For the full 4x4x4, each of the 16 intermediate values is formed from
    # a specific row of A and a specific column of B.
    #
    # The 40 multiplications pair up (fA[i], fB[j]) for specific (i,j).
    #
    # Actually, let me use the known F(4) factorization.
    # The F(4) transform for 4x4 matrix mult has:
    # - 16 transforms of A rows
    # - 16 transforms of B columns  
    # - 40 scalar multiplications
    # - 16 transforms of results to get C
    #
    # For tensor decomposition, each scalar multiplication gives a rank-1 term.
    #
    # Let me use a cleaner approach. The F(4) scheme:
    # A' = F_A * A (where F_A is 16x16, built from 16 row transforms applied to each row)
    # B' = B * F_B^T (where F_B is 16x16, built from 16 col transforms applied to each col)
    # C' = A' * B' (element-wise, 16x16 = 256 terms, but only 40 are non-zero)
    # C = F_C * C' (F_C is 16x16)
    #
    # This is getting complex. Let me just use the verified 49+16=65 approach but
    # try to find a better 4x4x4 decomposition.
    #
    # Actually, let me just carefully implement the known rank-49 Strassen for 4x4x4
    # plus rank-16 schoolbook for 4x4x1, giving total rank 65.
    # This is what the current program does and it scores 1.2308.
    #
    # Let me try a different split: 4x5x5 won't help.
    # What about treating it as 4x4x5 directly?
    # Schoolbook 4x4x5 = 4*4*5 = 80 terms.
    #
    # Let me try: split A into two 2x4 blocks.
    # A = [A_top; A_bottom] where A_top is 2x4, A_bottom is 2x4
    # C_top = A_top * B (2x4 * 4x5 = 2x5, rank 2*4*5=40 schoolbook)
    # C_bottom = A_bottom * B (2x4 * 4x5 = 2x5, rank 40 schoolbook)
    # Total: 80. Worse.
    #
    # What about Strassen for 2x4x5?
    # 2x4x5: split into 2x2x2 + 2x2x3
    # 2x2x2: Strassen rank 7
    # 2x2x3: schoolbook rank 12
    # Total per half: 7+12 = 19
    # Two halves: 38
    # Hmm, but this doesn't quite work because the 2x2x3 part needs care.
    #
    # Actually let me try: A is 4x4, B is 4x5.
    # Split B into B1 (4x4) and B2 (4x1).
    # C1 = A*B1 (4x4x4), C2 = A*B2 (4x4x1)
    #
    # For 4x4x4: use rank 49 (double Strassen) or rank 40 (F(4))
    # For 4x4x1: use rank 16 (schoolbook) or potentially less
    #
    # Let me try to get 4x4x1 with less than 16.
    # 4x4x1: C[i] = sum_j A[i,j]*B[j,0]
    # This is a bilinear form. The tensor rank of 4x4x1 is 4*4*1 = 16 for schoolbook.
    # Can we do better? The matrix multiplication tensor [4,4,1] has rank 16 (it's
    # essentially a 4x4 matrix times a vector, and the rank is 16 since each of the
    # 16 products is independent).
    #
    # So minimum for 4x4x1 is 16.
    #
    # For 4x4x4, the known best is 40 (Winograd F(4)).
    # Total: 40 + 16 = 56.
    #
    # Let me implement F(4) carefully.

    # Reset
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # F(4) Winograd transform for 4x4x4
    # Reference: Winograd, "Arithmetic Complexity of Computer Systems", 1984
    #
    # The F(4) scheme:
    # 16 intermediate values from A: a_i = sum_j F_A[i,j] * A[row_i, j]
    # 16 intermediate values from B: b_i = sum_j F_B[i,j] * B[j, col_i]
    # 40 products: p_k = a_{i_k} * b_{j_k}
    # 16 output values from C: C[row_k, col_k] = sum_l F_C[k,l] * p_{m_k,l}
    #
    # Actually, the standard F(4) has:
    # - 4x4 A and 4x4 B produce 4x4 C
    # - 16 "transformed" A values, 16 "transformed" B values
    # - 40 scalar multiplications
    # - 16 "inverse transformed" C values
    #
    # The specific F(4) factorization (from Winograd):
    #
    # Row transforms for A (each row of A gets transformed):
    # The 16 A-transforms are applied per-row. For each row r of A:
    #   a_{r,0} = A[r,0]
    #   a_{r,1} = A[r,3]  
    #   a_{r,2} = A[r,0]+A[r,3]
    #   a_{r,3} = A[r,1]-A[r,2]
    #   a_{r,4} = A[r,1]+A[r,2]
    #   a_{r,5} = A[r,1]+A[r,2]
    #   a_{r,6} = A[r,1]+A[r,2]
    #   a_{r,7} = A[r,1]
    #   a_{r,8} = A[r,1]
    #   a_{r,9} = A[r,1]
    #   a_{r,10} = A[r,2]
    #   a_{r,11} = A[r,2]
    #   a_{r,12} = A[r,2]
    #   a_{r,13} = A[r,0]-A[r,3]
    #   a_{r,14} = A[r,0]-A[r,3]
    #   a_{r,15} = A[r,0]-A[r,3]
    #
    # Similarly for B columns.
    #
    # The 40 pairings are specific. Let me use the known pairing table.
    #
    # Actually, this is getting very complex to get right. Let me use a verified
    # approach: I'll use the double-Strassen (rank 49) for 4x4x4 which I know works,
    # plus schoolbook (rank 16) for 4x4x1.
    #
    # But wait - the current program already does this and gets 65.
    # Let me try to actually get F(4) right.

    # F(4) pairing table (which A-transform pairs with which B-transform):
    # From Winograd's F(4):
    # The 40 products are:
    #
    # Let me define it differently. I'll use the known 40-term F(4) scheme.
    #
    # A-transform vectors (16 vectors, each in R^4):
    fA_vecs = [
        (1, 0, 0, 0),    # 0
        (0, 0, 0, 1),    # 1
        (1, 0, 0, 1),    # 2
        (0, 1, -1, 0),   # 3
        (0, 1, 1, 0),    # 4
        (0, 1, 1, 0),    # 5
        (0, 1, 1, 0),    # 6
        (0, 1, 0, 0),    # 7
        (0, 1, 0, 0),    # 8
        (0, 1, 0, 0),    # 9
        (0, 0, 1, 0),    # 10
        (0, 0, 1, 0),    # 11
        (0, 0, 1, 0),    # 12
        (1, 0, 0, -1),   # 13
        (1, 0, 0, -1),   # 14
        (1, 0, 0, -1),   # 15
    ]

    fB_vecs = [
        (1, 0, 0, 0),    # 0
        (0, 0, 0, 1),    # 1
        (1, 0, 0, 1),    # 2
        (0, 1, -1, 0),   # 3
        (0, 1, 1, 0),    # 4
        (0, 1, 1, 0),    # 5
        (0, 1, 1, 0),    # 6
        (0, 1, 0, 0),    # 7
        (0, 1, 0, 0),    # 8
        (0, 1, 0, 0),    # 9
        (0, 0, 1, 0),    # 10
        (0, 0, 1, 0),    # 11
        (0, 0, 1, 0),    # 12
        (1, 0, 0, -1),   # 13
        (1, 0, 0, -1),   # 14
        (1, 0, 0, -1),   # 15
    ]

    # The 40 pairings (a_idx, b_idx) for F(4):
    pairings = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2),
        (3, 3), (3, 4), (3, 5),
        (4, 3), (4, 4), (4, 5),
        (5, 3), (5, 4), (5, 5),
        (6, 3), (6, 4), (6, 5),
        (7, 0), (7, 1), (7, 2),
        (8, 0), (8, 1), (8, 2),
        (9, 0), (9, 1), (9, 2),
        (10, 3), (10, 4), (10, 5),
        (11, 3), (11, 4), (11, 5),
        (12, 3), (12, 4), (12, 5),
        (13, 0), (13, 1), (13, 2),
        (14, 0), (14, 1), (14, 2),
        (15, 0), (15, 1), (15, 2),
    ]

    # That's 48, not 40. F(4) has exactly 40 multiplications.
    # The pairing isn't all-to-all. Let me look this up properly.
    #
    # The F(4) scheme has:
    # - 3 groups of size 3x3 = 27 from the "diagonal" part
    # - 13 more from the "off-diagonal" part
    # Total: 40
    #
    # Actually I think the standard F(4) pairings are:
    # Group 1 (indices 0,1,2 for A and 0,1,2 for B): 3x3 = 9
    # Group 2 (indices 3,4,5 for A and 3,4,5 for B): 3x3 = 9
    # Group 3 (indices 7,8,9 for A and 0,1,2 for B): 3x3 = 9
    # Group 4 (indices 10,11,12 for A and 3,4,5 for B): 3x3 = 9
    # Group 5 (indices 13,14,15 for A and 0,1,2 for B): 3x3 = 9
    # That's 45. Still not 40.
    #
    # I'm not getting the F(4) right from memory. Let me just use the
    # verified rank-49 approach.

    # RESET - use rank 49 + 16 = 65
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Double Strassen for 4x4x4 (rank 49)
    # Split 4x4 into four 2x2 blocks
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B
    #
    # Strassen 2x2x2 (rank 7) applied at the 2x2 block level,
    # but we need 4x4x4 so we apply Strassen twice.
    #
    # Level 1: 7 multiplications of 2x2 matrices
    # Level 2: each 2x2 multiplication uses 7 scalar mults
    # Total: 7*7 = 49

    # Strassen 2x2x2 inner definitions:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    inner_defs = [
        # (A_entries: list of (row, col, coeff), B_entries, C_contributions: list of (row, col, coeff))
        ([((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))]),       # M1
        ([((1,0,1),(1,1,1))], [((0,0,1))], [((1,0,1),(1,1,-1))]),              # M2
        ([((0,0,1))], [((1,0,1),(1,1,-1))], [((0,1,1),(1,1,1))]),              # M3
        ([((1,1,1))], [((0,1,1),(1,0,-1))], [((0,0,1),(1,0,1))]),              # M4
        ([((0,0,1),(0,1,1))], [((1,1,1))], [((0,0,-1),(0,1,1))]),              # M5
        ([((1,0,1),(0,0,-1))], [((0,0,1),(0,1,1))], [((1,1,1))]),              # M6
        ([((0,1,1),(1,1,-1))], [((1,0,1),(1,1,1))], [((0,0,1))]),              # M7
    ]

    # Super Strassen: combine two levels
    # For 4x4x4, we split into 2x2 blocks of 2x2.
    # Block (br, bc) refers to block row br, block col bc (each 0 or 1)
    # Within each block, local row lr, local col lc (each 0 or 1)
    # Global row = br*2 + lr, global col = bc*2 + lc

    # Level 1 Strassen gives 7 "block" multiplications:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0]-B[1,1])
    # M4 = A[1,1] * (B[0,1]-B[1,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])

    # Where A[br,bc] is the 2x2 block at block position (br,bc)
    # A[0,0] = rows 0-1, cols 0-1
    # A[0,1] = rows 0-1, cols 2-3
    # A[1,0] = rows 2-3, cols 0-1
    # A[1,1] = rows 2-3, cols 2-3

    # Super Strassen definitions:
    # (A_block_coeffs: dict from block_idx to coeff, B_block_coeffs, C_block_contribs)
    # block_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)

    super_defs = [
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: A00(B10-B11) -> C01 += M3, C11 += M3
        ({0: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: A11(B01-B10) -> C00 += M4, C10 += M4
        ({3: 1}, {1: 1, 2: -1}, {0: 1, 2: 1}),
        # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (A01-A11)(B10+B11) -> C00 += M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    def block_to_global_row(br, bc):
        # Returns list of (global_row, local_row) for a block
        # Block (br, bc) covers rows br*2..br*2+1, cols bc*2..bc*2+1
        pass

    for s_idx, (a_blocks, b_blocks, c_blocks) in enumerate(super_defs):
        for i_idx, (a_local, b_local, c_local) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []

            # A part: for each block with coeff, for each local entry
            for block_idx, block_coeff in a_blocks.items():
                br = block_idx // 2
                bc = block_idx % 2
                for (lr, lc, local_coeff) in a_local:
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    total_coeff = block_coeff * local_coeff
                    if total_coeff != 0:
                        u_list.append((gr, gc, total_coeff))

            # B part
            for block_idx, block_coeff in b_blocks.items():
                br = block_idx // 2
                bc = block_idx % 2
                for (lr, lc, local_coeff) in b_local:
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    total_coeff = block_coeff * local_coeff
                    if total_coeff != 0:
                        v_list.append((gr, gc, total_coeff))

            # C part
            for block_idx, block_coeff in c_blocks.items():
                br = block_idx // 2
                bc = block_idx % 2
                for (lr, lc, local_coeff) in c_local:
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    total_coeff = block_coeff * local_coeff
                    if total_coeff != 0:
                        w_list.append((gc, gr, total_coeff))  # W is indexed [k*n+i] = [col*4+row]

            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    # Part 2: 4x4x1 (rank 16) - schoolbook for the 5th column
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1  # W[k*n+i] where k=4 (5th col), i=row
            col += 1

    if time.time() - start_time > DEADLINE - 5:
        pass

    return {"U": U, "V": V, "W": W}
