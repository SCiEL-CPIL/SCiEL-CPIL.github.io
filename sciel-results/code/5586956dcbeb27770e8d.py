"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    n, m, p = 4, 4, 7

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_dict, v_dict, w_dict):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        for idx, coeff in u_dict.items():
            if coeff != 0:
                u_col[idx] = coeff
        for idx, coeff in v_dict.items():
            if coeff != 0:
                v_col[idx] = coeff
        for idx, coeff in w_dict.items():
            if coeff != 0:
                w_col[idx] = coeff
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def a_idx(i, j):
        return i * m + j

    def b_idx(j, k):
        return j * p + k

    def c_idx(i, k):
        return k * n + i

    def strassen_2x2(va, vb, vc):
        """
        Apply Strassen to a virtual 2x2 x 2x2 multiplication.
        va[i][j] = dict of {a_index: coeff} for virtual A entry (i,j)
        vb[i][j] = dict of {b_index: coeff} for virtual B entry (i,j)
        vc[i][j] = dict of {c_index: coeff} for virtual C entry (i,j)
        """
        # M1 = (A00+A11)(B00+B11)
        u1 = {}
        for i in range(2):
            for j in range(2):
                if (i, j) in [(0, 0), (1, 1)]:
                    for k, v in va[i][j].items():
                        u1[k] = u1.get(k, 0) + v
        v1 = {}
        for i in range(2):
            for j in range(2):
                if (i, j) in [(0, 0), (1, 1)]:
                    for k, v in vb[i][j].items():
                        v1[k] = v1.get(k, 0) + v
        w1 = {}
        for i in range(2):
            for j in range(2):
                if (i, j) in [(0, 0), (1, 1)]:
                    for k, v in vc[i][j].items():
                        w1[k] = w1.get(k, 0) + v
        add_term(u1, v1, w1)

        # M2 = (A10+A11)B00
        u2 = {}
        for k, v in va[1][0].items():
            u2[k] = u2.get(k, 0) + v
        for k, v in va[1][1].items():
            u2[k] = u2.get(k, 0) + v
        v2 = dict(vb[0][0])
        w2 = {}
        for k, v in vc[1][0].items():
            w2[k] = w2.get(k, 0) + v
        for k, v in vc[1][1].items():
            w2[k] = w2.get(k, 0) - v
        add_term(u2, v2, w2)

        # M3 = A00(B01-B11)
        u3 = dict(va[0][0])
        v3 = {}
        for k, v in vb[0][1].items():
            v3[k] = v3.get(k, 0) + v
        for k, v in vb[1][1].items():
            v3[k] = v3.get(k, 0) - v
        w3 = {}
        for k, v in vc[0][1].items():
            w3[k] = w3.get(k, 0) + v
        for k, v in vc[0][0].items():
            w3[k] = w3.get(k, 0) + v
        add_term(u3, v3, w3)

        # M4 = A11(B10-B00)
        u4 = dict(va[1][1])
        v4 = {}
        for k, v in vb[1][0].items():
            v4[k] = v4.get(k, 0) + v
        for k, v in vb[0][0].items():
            v4[k] = v4.get(k, 0) - v
        w4 = {}
        for k, v in vc[0][0].items():
            w4[k] = w4.get(k, 0) + v
        for k, v in vc[1][0].items():
            w4[k] = w4.get(k, 0) + v
        add_term(u4, v4, w4)

        # M5 = (A00+A01)B11
        u5 = {}
        for k, v in va[0][0].items():
            u5[k] = u5.get(k, 0) + v
        for k, v in va[0][1].items():
            u5[k] = u5.get(k, 0) + v
        v5 = dict(vb[1][1])
        w5 = {}
        for k, v in vc[0][0].items():
            w5[k] = w5.get(k, 0) - v
        for k, v in vc[0][1].items():
            w5[k] = w5.get(k, 0) + v
        add_term(u5, v5, w5)

        # M6 = (A10-A00)(B00+B01)
        u6 = {}
        for k, v in va[1][0].items():
            u6[k] = u6.get(k, 0) + v
        for k, v in va[0][0].items():
            u6[k] = u6.get(k, 0) - v
        v6 = {}
        for k, v in vb[0][0].items():
            v6[k] = v6.get(k, 0) + v
        for k, v in vb[0][1].items():
            v6[k] = v6.get(k, 0) + v
        w6 = dict(vc[1][1])
        add_term(u6, v6, w6)

        # M7 = (A01-A11)(B10+B11)
        u7 = {}
        for k, v in va[0][1].items():
            u7[k] = u7.get(k, 0) + v
        for k, v in va[1][1].items():
            u7[k] = u7.get(k, 0) - v
        v7 = {}
        for k, v in vb[1][0].items():
            v7[k] = v7.get(k, 0) + v
        for k, v in vb[1][1].items():
            v7[k] = v7.get(k, 0) + v
        w7 = dict(vc[0][0])
        add_term(u7, v7, w7)

    def schoolbook_2x2_2x1(a_rows, a_cols, b_row, b_col, c_row, c_col):
        """Add 4 rank-1 terms for 2x2 x 2x1 schoolbook."""
        for i in range(2):
            for j in range(2):
                add_term(
                    {a_idx(a_rows[i], a_cols[j]): 1},
                    {b_idx(b_row, b_col): 1},
                    {c_idx(c_row[i], c_col): 1}
                )

    # Split the problem:
    # A is 4x4, B is 4x7
    # Split B into B1 (4x4, cols 0-3) and B2 (4x3, cols 4-6)
    # C = A*B1 + A*B2
    # C is 4x7

    # For A*B1 (4x4 x 4x4 -> 4x4), use Strassen
    # For A*B2 (4x4 x 4x3 -> 4x3), use schoolbook or split further

    # Let's split B2 into B2a (4x2, cols 4-5) and B2b (4x1, col 6)
    # A*B2a: 4x4 x 4x2 -> 4x2
    # A*B2b: 4x4 x 4x1 -> 4x1

    # For 4x4 x 4x2: split into two 4x4 x 4x1
    # For each 4x4 x 4x1: schoolbook = 16 mults
    # Total for B2: 2*16 + 16 = 48 mults (schoolbook)
    # Or we can be smarter.

    # Actually, let me use a different split.
    # B is 4x7. Split into B1 (4x4) and B2 (4x3).
    # A*B1: Strassen 4x4x4x4 = 49 mults
    # A*B2: 4x4 x 4x3. Split B2 into three 4x1 columns.
    # Each 4x4 x 4x1 = 16 mults schoolbook.
    # Total: 49 + 3*16 = 49 + 48 = 97 mults.

    # Better: split B2 (4x3) into B2a (4x2) and B2b (4x1).
    # A*B2a (4x4 x 4x2): split into two 4x4 x 4x1 = 32 mults
    # A*B2b (4x4 x 4x1): 16 mults
    # Total: 49 + 32 + 16 = 97. Same.

    # Even better: use Strassen-like for 4x4 x 4x2?
    # 4x4 x 4x2 can be seen as computing two 4x4 x 4x1 products.
    # Not much to gain there.

    # Alternative: split A into 2x2 blocks and use block Strassen for the full 4x4 x 4x7.
    # Let me try a different approach: treat the whole thing as block matrices.

    # A is 4x4, split into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # B is 4x7, split into 2x2 blocks horizontally: B1, B2, B3, B4
    #   B1: rows 0-1, cols 0-1 (2x2)
    #   B2: rows 0-1, cols 2-3 (2x2)
    #   B3: rows 0-1, cols 4-5 (2x2)
    #   B4: rows 0-1, col 6 (2x1)
    #   B5: rows 2-3, cols 0-1 (2x2)
    #   B6: rows 2-3, cols 2-3 (2x2)
    #   B7: rows 2-3, cols 4-5 (2x2)
    #   B8: rows 2-3, col 6 (2x1)

    # C is 4x7, split into 2x2 blocks:
    #   C1: rows 0-1, cols 0-1 (2x2)
    #   C2: rows 0-1, cols 2-3 (2x2)
    #   C3: rows 0-1, cols 4-5 (2x2)
    #   C4: rows 0-1, col 6 (2x1)
    #   C5: rows 2-3, cols 0-1 (2x2)
    #   C6: rows 2-3, cols 2-3 (2x2)
    #   C7: rows 2-3, cols 4-5 (2x2)
    #   C8: rows 2-3, col 6 (2x1)

    # C1 = A11*B1 + A12*B5
    # C2 = A11*B2 + A12*B6
    # C3 = A11*B3 + A12*B7
    # C4 = A11*B4 + A12*B8
    # C5 = A21*B1 + A22*B5
    # C6 = A21*B2 + A22*B6
    # C7 = A21*B3 + A22*B7
    # C8 = A21*B4 + A22*B8

    # Each 2x2 x 2x2 = 4 mults schoolbook or 7 mults Strassen.
    # Each 2x2 x 2x1 = 4 mults schoolbook.

    # If we use Strassen for all 2x2x2x2 products:
    # There are 12 such products (4 C-blocks that are 2x2 from A11, 4 from A12, 4 from A21, 4 from A22... wait)
    # C1: A11*B1 (2x2x2x2) + A12*B5 (2x2x2x2) = 2 Strassen = 14
    # C2: A11*B2 + A12*B6 = 14
    # C3: A11*B3 + A12*B7 = 14
    # C4: A11*B4 + A12*B8 = 2 schoolbook 2x2x2x1 = 8
    # C5: A21*B1 + A22*B5 = 14
    # C6: A21*B2 + A22*B6 = 14
    # C7: A21*B3 + A22*B7 = 14
    # C8: A21*B4 + A22*B8 = 8
    # Total: 7*14 + 2*8 = 98 + 16 = 114. Worse than schoolbook 112.

    # Let me just use the straightforward approach:
    # A*B1 (4x4 x 4x4) with Strassen: 49 mults
    # A*B2 (4x4 x 4x3) with schoolbook: 48 mults
    # Total: 97 mults

    # Let's implement this.

    # Part 1: A * B1 where B1 = B[:, 0:4]
    # Use Strassen on 4x4 x 4x4.

    # Define virtual 2x2 blocks for Strassen at 4x4 level.
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B1_11 = B[0:2, 0:2], B1_12 = B[0:2, 2:4], B1_21 = B[2:4, 0:2], B1_22 = B[2:4, 2:4]
    # C1_11 = C[0:2, 0:2], C1_12 = C[0:2, 2:4], C1_21 = C[2:4, 0:2], C1_22 = C[2:4, 2:4]

    # Strassen combinations for 4x4x4x4:
    # M1 = (A11+A22)(B1_11+B1_22)
    # M2 = (A21+A22)B1_11
    # M3 = A11(B1_12-B1_22)
    # M4 = A22(B1_21-B1_11)
    # M5 = (A11+A12)B1_22
    # M6 = (A21-A11)(B1_11+B1_12)
    # M7 = (A12-A22)(B1_21+B1_22)

    # C1_11 = M1 + M4 - M5 + M7
    # C1_12 = M3 + M5
    # C1_21 = M2 + M4
    # C1_22 = M1 - M2 + M3 + M6

    # Each Mi is a 2x2 x 2x2 product, done with Strassen (7 mults).

    # Let me define the virtual matrices for each Mi.

    # Helper to create a virtual 2x2 matrix from linear combinations
    def make_va(a_block_list):
        """a_block_list: list of 4 dicts, for positions (0,0), (0,1), (1,0), (1,1)"""
        return [
            [a_block_list[0], a_block_list[1]],
            [a_block_list[2], a_block_list[3]]
        ]

    def make_vb(b_block_list):
        return [
            [b_block_list[0], b_block_list[1]],
            [b_block_list[2], b_block_list[3]]
        ]

    def make_vc(c_block_list):
        return [
            [c_block_list[0], c_block_list[1]],
            [c_block_list[2], c_block_list[3]]
        ]

    # A block indices
    A11 = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (i,j) pairs for A[0:2, 0:2]
    A12 = [(0, 2), (0, 3), (1, 2), (1, 3)]  # A[0:2, 2:4]
    A21 = [(2, 0), (2, 1), (3, 0), (3, 1)]  # A[2:4, 0:2]
    A22 = [(2, 2), (2, 3), (3, 2), (3, 3)]  # A[2:4, 2:4]

    # B1 block indices (B columns 0-3)
    B1_11 = [(0, 0), (0, 1), (1, 0), (1, 1)]  # B[0:2, 0:2]
    B1_12 = [(0, 2), (0, 3), (1, 2), (1, 3)]  # B[0:2, 2:4]
    B1_21 = [(2, 0), (2, 1), (3, 0), (3, 1)]  # B[2:4, 0:2]
    B1_22 = [(2, 2), (2, 3), (3, 2), (3, 3)]  # B[2:4, 2:4]

    # C1 block indices (C columns 0-3)
    C1_11 = [(0, 0), (0, 1), (1, 0), (1, 1)]  # C[0:2, 0:2]
    C1_12 = [(0, 2), (0, 3), (1, 2), (1, 3)]  # C[0:2, 2:4]
    C1_21 = [(2, 0), (2, 1), (3, 0), (3, 1)]  # C[2:4, 0:2]
    C1_22 = [(2, 2), (2, 3), (3, 2), (3, 3)]  # C[2:4, 2:4]

    def block_to_dict(idx_list, sign=1):
        """Convert a list of (i,j) pairs to a dict {a_idx: coeff}"""
        d = {}
        for (i, j) in idx_list:
            d[a_idx(i, j)] = sign
        return d

    def block_to_dict_b(idx_list, sign=1):
        d = {}
        for (i, j) in idx_list:
            d[b_idx(i, j)] = sign
        return d

    def block_to_dict_c(idx_list, sign=1):
        d = {}
        for (i, j) in idx_list:
            d[c_idx(i, j)] = sign
        return d

    # For each Strassen Mi, define the virtual 2x2 A, B, C matrices.
    # Each entry is a dict of original indices with coefficients.

    # M1 = (A11+A22)(B1_11+B1_22)
    va_M1 = make_va([
        block_to_dict(A11[:1]) | block_to_dict(A22[:1]),  # (0,0): A[0,0]+A[2,2]
        block_to_dict(A11[1:2]) | block_to_dict(A22[1:2]),  # (0,1): A[0,1]+A[2,3]
        block_to_dict(A11[2:3]) | block_to_dict(A22[2:3]),  # (1,0): A[1,0]+A[3,2]
        block_to_dict(A11[3:4]) | block_to_dict(A22[3:4])   # (1,1): A[1,1]+A[3,3]
    ])
    vb_M1 = make_vb([
        block_to_dict_b(B1_11[:1]) | block_to_dict_b(B1_22[:1]),
        block_to_dict_b(B1_11[1:2]) | block_to_dict_b(B1_22[1:2]),
        block_to_dict_b(B1_11[2:3]) | block_to_dict_b(B1_22[2:3]),
        block_to_dict_b(B1_11[3:4]) | block_to_dict_b(B1_22[3:4])
    ])
    # C contribution: C1_11 += M1, C1_22 += M1
    vc_M1 = make_vc([
        block_to_dict_c(C1_11[:1]) | block_to_dict_c(C1_22[:1]),  # (0,0)
        block_to_dict_c(C1_11[1:2]) | block_to_dict_c(C1_22[1:2]),  # (0,1)
        block_to_dict_c(C1_11[2:3]) | block_to_dict_c(C1_22[2:3]),  # (1,0)
        block_to_dict_c(C1_11[3:4]) | block_to_dict_c(C1_22[3:4])   # (1,1)
    ])
    # Wait, this is wrong. The virtual C for M1 should have the same structure as the 2x2 output.
    # M1 is a 2x2 matrix. It contributes to C1_11 and C1_22.
    # So the virtual C matrix for M1 has:
    #   vc[0][0] = C1_11[0][0] contribution
    #   vc[0][1] = C1_11[0][1] contribution
    #   vc[1][0] = C1_11[1][0] contribution
    #   vc[1][1] = C1_11[1][1] contribution
    # And separately, M1 also contributes to C1_22 with the same values.
    # But in my strassen_2x2 function, vc is a single 2x2 matrix.
    # I need to handle the fact that M1 contributes to TWO different 2x2 blocks.

    # Let me restructure. Instead of one strassen_2x2 call per Mi, I'll make the vc
    # contain the combined contributions.

    # Actually, the cleanest way: for each Mi, I define what it contributes to.
    # M1 contributes to C1_11 (all 4 entries) and C1_22 (all 4 entries).
    # So vc_M1 should be a 2x2 where each entry is a dict that includes both C1_11 and C1_22 contributions.

    # Let me redo this more carefully.

    # Clear and restart with a cleaner approach.
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()

    def add_strassen_term(va, vb, vc):
        """
        va: 2x2 list of dicts {a_idx: coeff}
        vb: 2x2 list of dicts {b_idx: coeff}
        vc: 2x2 list of dicts {c_idx: coeff}
        """
        # M1 = (A00+A11)(B00+B11)
        u1 = {}
        for i in [0, 1]:
            for j in [0, 1]:
                if (i == 0 and j == 0) or (i == 1 and j == 1):
                    for k, v in va[i][j].items():
                        u1[k] = u1.get(k, 0) + v
        v1 = {}
        for i in [0, 1]:
            for j in [0, 1]:
                if (i == 0 and j == 0) or (i == 1 and j == 1):
                    for k, v in vb[i][j].items():
                        v1[k] = v1.get(k, 0) + v
        w1 = {}
        for i in [0, 1]:
            for j in [0, 1]:
                if (i == 0 and j == 0) or (i == 1 and j == 1):
                    for k, v in vc[i][j].items():
                        w1[k] = w1.get(k, 0) + v
        add_term(u1, v1, w1)

        # M2 = (A10+A11)B00
        u2 = {}
        for k, v in va[1][0].items():
            u2[k] = u2.get(k, 0) + v
        for k, v in va[1][1].items():
            u2[k] = u2.get(k, 0) + v
        v2 = dict(vb[0][0])
        w2 = {}
        for k, v in vc[1][0].items():
            w2[k] = w2.get(k, 0) + v
        for k, v in vc[1][1].items():
            w2[k] = w2.get(k, 0) - v
        add_term(u2, v2, w2)

        # M3 = A00(B01-B11)
        u3 = dict(va[0][0])
        v3 = {}
        for k, v in vb[0][1].items():
            v3[k] = v3.get(k, 0) + v
        for k, v in vb[1][1].items():
            v3[k] = v3.get(k, 0) - v
        w3 = {}
        for k, v in vc[0][1].items():
            w3[k] = w3.get(k, 0) + v
        for k, v in vc[0][0].items():
            w3[k] = w3.get(k, 0) + v
        add_term(u3, v3, w3)

        # M4 = A11(B10-B00)
        u4 = dict(va[1][1])
        v4 = {}
        for k, v in vb[1][0].items():
            v4[k] = v4.get(k, 0) + v
        for k, v in vb[0][0].items():
            v4[k] = v4.get(k, 0) - v
        w4 = {}
        for k, v in vc[0][0].items():
            w4[k] = w4.get(k, 0) + v
        for k, v in vc[1][0].items():
            w4[k] = w4.get(k, 0) + v
        add_term(u4, v4, w4)

        # M5 = (A00+A01)B11
        u5 = {}
        for k, v in va[0][0].items():
            u5[k] = u5.get(k, 0) + v
        for k, v in va[0][1].items():
            u5[k] = u5.get(k, 0) + v
        v5 = dict(vb[1][1])
        w5 = {}
        for k, v in vc[0][0].items():
            w5[k] = w5.get(k, 0) - v
        for k, v in vc[0][1].items():
            w5[k] = w5.get(k, 0) + v
        add_term(u5, v5, w5)

        # M6 = (A10-A00)(B00+B01)
        u6 = {}
        for k, v in va[1][0].items():
            u6[k] = u6.get(k, 0) + v
        for k, v in va[0][0].items():
            u6[k] = u6.get(k, 0) - v
        v6 = {}
        for k, v in vb[0][0].items():
            v6[k] = v6.get(k, 0) + v
        for k, v in vb[0][1].items():
            v6[k] = v6.get(k, 0) + v
        w6 = dict(vc[1][1])
        add_term(u6, v6, w6)

        # M7 = (A01-A11)(B10+B11)
        u7 = {}
        for k, v in va[0][1].items():
            u7[k] = u7.get(k, 0) + v
        for k, v in va[1][1].items():
            u7[k] = u7.get(k, 0) - v
        v7 = {}
        for k, v in vb[1][0].items():
            v7[k] = v7.get(k, 0) + v
        for k, v in vb[1][1].items():
            v7[k] = v7.get(k, 0) + v
        w7 = dict(vc[0][0])
        add_term(u7, v7, w7)

    # Now define the 7 Strassen combinations for 4x4x4x4.
    # Each Mi is a 2x2x2x2 product. The "A" and "B" are 2x2 virtual matrices
    # whose entries are linear combinations of the original A and B entries.
    # The "C" is a 2x2 virtual matrix whose entries are linear combinations of C entries.

    # For M1 = (A11+A22)(B1_11+B1_22):
    # va[0][0] = {A[0,0]:1, A[2,2]:1}
    # va[0][1] = {A[0,1]:1, A[2,3]:1}
    # va[1][0] = {A[1,0]:1, A[3,2]:1}
    # va[1][1] = {A[1,1]:1, A[3,3]:1}
    # vb[0][0] = {B[0,0]:1, B[2,2]:1}
    # vb[0][1] = {B[0,1]:1, B[2,3]:1}
    # vb[1][0] = {B[1,0]:1, B[3,2]:1}
    # vb[1][1] = {B[1,1]:1, B[3,3]:1}
    # vc[0][0] = {C[0,0]:1, C[2,2]:1}  (M1 contributes to C1_11[0,0] and C1_22[0,0])
    # vc[0][1] = {C[0,1]:1, C[2,3]:1}
    # vc[1][0] = {C[1,0]:1, C[3,2]:1}
    # vc[1][1] = {C[1,1]:1, C[3,3]:1}

    def A_dict(i, j, s=1):
        return {a_idx(i, j): s}

    def B_dict(j, k, s=1):
        return {b_idx(j, k): s}

    def C_dict(i, k, s=1):
        return {c_idx(i, k): s}

    def merge_dict(d1, d2):
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) + v
        return result

    # M1 = (A11+A22)(B1_11+B1_22)
    va_M1 = [
        [merge_dict(A_dict(0,0), A_dict(2,2)), merge_dict(A_dict(0,1), A_dict(2,3))],
        [merge_dict(A_dict(1,0), A_dict(3,2)), merge_dict(A_dict(1,1), A_dict(3,3))]
    ]
    vb_M1 = [
        [merge_dict(B_dict(0,0), B_dict(2,2)), merge_dict(B_dict(0,1), B_dict(2,3))],
        [merge_dict(B_dict(1,0), B_dict(3,2)), merge_dict(B_dict(1,1), B_dict(3,3))]
    ]
    vc_M1 = [
        [merge_dict(C_dict(0,0), C_dict(2,2)), merge_dict(C_dict(0,1), C_dict(2,3))],
        [merge_dict(C_dict(1,0), C_dict(3,2)), merge_dict(C_dict(1,1), C_dict(3,3))]
    ]
    add_strassen_term(va_M1, vb_M1, vc_M1)

    # M2 = (A21+A22)B1_11
    va_M2 = [
        [A_dict(0,0), A_dict(0,1)],
        [merge_dict(A_dict(2,0), A_dict(2,2)), merge_dict(A_dict(2,1), A_dict(2,3))]
    ]
    # Wait, A21+A22 means:
    # (A21+A22)[0][0] = A[2,0] + A[2,2]
    # (A21+A22)[0][1] = A[2,1] + A[2,3]
    # (A21+A22)[1][0] = A[3,0] + A[3,2]
    # (A21+A22)[1][1] = A[3,1] + A[3,3]
    va_M2 = [
        [A_dict(0,0), A_dict(0,1)],
        [merge_dict(A_dict(2,0), A_dict(2,2)), merge_dict(A_dict(2,1), A_dict(2,3))]
    ]
    # Hmm, that's not right either. A21 is A[2:4, 0:2], A22 is A[2:4, 2:4].
    # (A21+A22) is a 2x2 matrix:
    #   [0][0] = A[2,0] + A[2,2]
    #   [0][1] = A[2,1] + A[2,3]
    #   [1][0] = A[3,0] + A[3,2]
    #   [1][1] = A[3,1] + A[3,3]
    va_M2 = [
        [merge_dict(A_dict(2,0), A_dict(2,2)), merge_dict(A_dict(2,1), A_dict(2,3))],
        [merge_dict(A_dict(3,0), A_dict(3,2)), merge_dict(A_dict(3,1), A_dict(3,3))]
    ]
    vb_M2 = [
        [B_dict(0,0), B_dict(0,1)],
        [B_dict(1,0), B_dict(1,1)]
    ]
    # M2 contributes to C1_21 and C1_22:
    # C1_21 += M2, C1_22 -= M2
    # vc[0][0] = C[2,0] (from C1_21[0,0])
    # vc[0][1] = C[2,1] (from C1_21[0,1])
    # vc[1][0] = C[3,0] (from C1_21[1,0])
    # vc[1][1] = C[3,1] (from C1_21[1,1])
    # And C1_22 -= M2 means:
    # vc[0][0] also has -C[2,2], vc[0][1] has -C[2,3], etc.
    vc_M2 = [
        [merge_dict(C_dict(2,0), C_dict(2,2,-1)), merge_dict(C_dict(2,1), C_dict(2,3,-1))],
        [merge_dict(C_dict(3,0), C_dict(3,2,-1)), merge_dict(C_dict(3,1), C_dict(3,3,-1))]
    ]
    add_strassen_term(va_M2, vb_M2, vc_M2)

    # M3 = A11(B1_12-B1_22)
    va_M3 = [
        [A_dict(0,0), A_dict(0,1)],
        [A_dict(1,0), A_dict(1,1)]
    ]
    vb_M3 = [
        [merge_dict(B_dict(0,2), B_dict(2,2,-1)), merge_dict(B_dict(0,3), B_dict(2,3,-1))],
        [merge_dict(B_dict(1,2), B_dict(3,2,-1)), merge_dict(B_dict(1,3), B_dict(3,3,-1))]
    ]
    # M3 contributes to C1_12 and C1_22:
    # C1_12 += M3, C1_22 += M3
    vc_M3 = [
        [merge_dict(C_dict(0,2), C_dict(2,2)), merge_dict(C_dict(0,3), C_dict(2,3))],
        [merge_dict(C_dict(1,2), C_dict(3,2)), merge_dict(C_dict(1,3), C_dict(3,3))]
    ]
    add_strassen_term(va_M3, vb_M3, vc_M3)

    # M4 = A22(B1_21-B1_11)
    va_M4 = [
        [A_dict(2,2), A_dict(2,3)],
        [A_dict(3,2), A_dict(3,3)]
    ]
    vb_M4 = [
        [merge_dict(B_dict(2,0), B_dict(0,0,-1)), merge_dict(B_dict(2,1), B_dict(0,1,-1))],
        [merge_dict(B_dict(3,0), B_dict(1,0,-1)), merge_dict(B_dict(3,1), B_dict(1,1,-1))]
    ]
    # M4 contributes to C1_11 and C1_21:
    # C1_11 += M4, C1_21 += M4
    vc_M4 = [
        [merge_dict(C_dict(0,0), C_dict(2,0)), merge_dict(C_dict(0,1), C_dict(2,1))],
        [merge_dict(C_dict(1,0), C_dict(3,0)), merge_dict(C_dict(1,1), C_dict(3,1))]
    ]
    add_strassen_term(va_M4, vb_M4, vc_M4)

    # M5 = (A11+A12)B1_22
    va_M5 = [
        [merge_dict(A_dict(0,0), A_dict(0,2)), merge_dict(A_dict(0,1), A_dict(0,3))],
        [merge_dict(A_dict(1,0), A_dict(1,2)), merge_dict(A_dict(1,1), A_dict(1,3))]
    ]
    vb_M5 = [
        [B_dict(2,2), B_dict(2,3)],
        [B_dict(3,2), B_dict(3,3)]
    ]
    # M5 contributes to C1_11 and C1_12:
    # C1_11 -= M5, C1_12 += M5
    vc_M5 = [
        [merge_dict(C_dict(0,0,-1), C_dict(0,2)), merge_dict(C_dict(0,1,-1), C_dict(0,3))],
        [merge_dict(C_dict(1,0,-1), C_dict(1,2)), merge_dict(C_dict(1,1,-1), C_dict(1,3))]
    ]
    add_strassen_term(va_M5, vb_M5, vc_M5)

    # M6 = (A21-A11)(B1_11+B1_12)
    va_M6 = [
        [merge_dict(A_dict(2,0), A_dict(0,0,-1)), merge_dict(A_dict(2,1), A_dict(0,1,-1))],
        [merge_dict(A_dict(3,0), A_dict(1,0,-1)), merge_dict(A_dict(3,1), A_dict(1,1,-1))]
    ]
    vb_M6 = [
        [merge_dict(B_dict(0,0), B_dict(0,2)), merge_dict(B_dict(0,1), B_dict(0,3))],
        [merge_dict(B_dict(1,0), B_dict(1,2)), merge_dict(B_dict(1,1), B_dict(1,3))]
    ]
    # M6 contributes to C1_22:
    # C1_22 += M6
    vc_M6 = [
        [C_dict(2,2), C_dict(2,3)],
        [C_dict(3,2), C_dict(3,3)]
    ]
    add_strassen_term(va_M6, vb_M6, vc_M6)

    # M7 = (A12-A22)(B1_21+B1_22)
    va_M7 = [
        [merge_dict(A_dict(0,2), A_dict(2,2,-1)), merge_dict(A_dict(0,3), A_dict(2,3,-1))],
        [merge_dict(A_dict(1,2), A_dict(3,2,-1)), merge_dict(A_dict(1,3), A_dict(3,3,-1))]
    ]
    vb_M7 = [
        [merge_dict(B_dict(2,0), B_dict(2,2)), merge_dict(B_dict(2,1), B_dict(2,3))],
        [merge_dict(B_dict(3,0), B_dict(3,2)), merge_dict(B_dict(3,1), B_dict(3,3))]
    ]
    # M7 contributes to C1_11:
    # C1_11 += M7
    vc_M7 = [
        [C_dict(0,0), C_dict(0,1)],
        [C_dict(1,0), C_dict(1,1)]
    ]
    add_strassen_term(va_M7, vb_M7, vc_M7)

    # Part 2: A * B2 where B2 = B[:, 4:7] (4x3 matrix)
    # Use schoolbook: 4*4*3 = 48 mults
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                add_term(
                    {a_idx(i, j): 1},
                    {b_idx(j, k): 1},
                    {c_idx(i, k): 1}
                )

    # Build the output matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for col in range(rank):
        for row in range(n * m):
            U[row][col] = U_cols[col][row]
        for row in range(m * p):
            V[row][col] = V_cols[col][row]
        for row in range(p * n):
            W[row][col] = W_cols[col][row]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
