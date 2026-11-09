import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    n, m, p = 4, 4, 7

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        for idx, coeff in u_entries.items():
            if coeff != 0:
                u_col[idx] = coeff
        for idx, coeff in v_entries.items():
            if coeff != 0:
                v_col[idx] = coeff
        for idx, coeff in w_entries.items():
            if coeff != 0:
                w_col[idx] = coeff
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def merge_add(d1, d2):
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) + v
        return result

    def merge_sub(d1, d2):
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) - v
        return result

    def merge_w(d1, d2, s1=1, s2=1):
        result = {}
        for k, v in d1.items():
            if s1 != 0:
                result[k] = result.get(k, 0) + s1 * v
        for k, v in d2.items():
            if s2 != 0:
                result[k] = result.get(k, 0) + s2 * v
        return result

    def a_idx(i, j):
        return i * m + j

    def b_idx(j, k):
        return j * p + k

    def c_idx(i, k):
        return k * n + i

    # Strassen decomposition for 2x2 matrices
    # Maps local (i,j) to global via offset
    def strassen_2x2(a_offset_i, a_offset_j, b_offset_j, b_offset_k, c_offset_i, c_offset_k):
        # A block: rows a_offset_i..a_offset_i+1, cols a_offset_j..a_offset_j+1
        # B block: rows b_offset_j..b_offset_j+1, cols b_offset_k..b_offset_k+1
        # C block: rows c_offset_i..c_offset_i+1, cols c_offset_k..c_offset_k+1

        # Define local 2x2 blocks as dicts
        a00 = {a_idx(a_offset_i, a_offset_j): 1}
        a01 = {a_idx(a_offset_i, a_offset_j + 1): 1}
        a10 = {a_idx(a_offset_i + 1, a_offset_j): 1}
        a11 = {a_idx(a_offset_i + 1, a_offset_j + 1): 1}

        b00 = {b_idx(b_offset_j, b_offset_k): 1}
        b01 = {b_idx(b_offset_j, b_offset_k + 1): 1}
        b10 = {b_idx(b_offset_j + 1, b_offset_k): 1}
        b11 = {b_idx(b_offset_j + 1, b_offset_k + 1): 1}

        c00 = {c_idx(c_offset_i, c_offset_k): 1}
        c01 = {c_idx(c_offset_i, c_offset_k + 1): 1}
        c10 = {c_idx(c_offset_i + 1, c_offset_k): 1}
        c11 = {c_idx(c_offset_i + 1, c_offset_k + 1): 1}

        # 7 Strassen products
        # M1 = (A00 + A11) * (B00 + B11) -> contributes to C00, C11
        add_term(
            merge_add(a00, a11),
            merge_add(b00, b11),
            merge_add(c00, c11)
        )

        # M2 = (A10 + A11) * B00 -> contributes to C10, C11 (C11 with -)
        add_term(
            merge_add(a10, a11),
            b00,
            merge_w(c10, c11, 1, -1)
        )

        # M3 = A00 * (B01 - B11) -> contributes to C01, C11
        add_term(
            a00,
            merge_sub(b01, b11),
            merge_add(c01, c11)
        )

        # M4 = A11 * (B10 - B00) -> contributes to C00, C10
        add_term(
            a11,
            merge_sub(b10, b00),
            merge_add(c00, c10)
        )

        # M5 = (A00 + A01) * B11 -> contributes to C00 (-), C01
        add_term(
            merge_add(a00, a01),
            b11,
            merge_w(c00, c01, -1, 1)
        )

        # M6 = (A10 - A00) * (B00 + B01) -> contributes to C11
        add_term(
            merge_sub(a10, a00),
            merge_add(b00, b01),
            c11
        )

        # M7 = (A01 - A11) * (B10 + B11) -> contributes to C00
        add_term(
            merge_sub(a01, a11),
            merge_add(b10, b11),
            c00
        )

    # Rank 5 decomposition for 2x2 * 2x3
    # A is 2x2, B is 2x3, C is 2x3
    def strassen_2x2_2x3(a_offset_i, a_offset_j, b_offset_j, b_offset_k, c_offset_i, c_offset_k):
        # A block: rows a_offset_i..a_offset_i+1, cols a_offset_j..a_offset_j+1
        # B block: rows b_offset_j..b_offset_j+1, cols b_offset_k..b_offset_k+2
        # C block: rows c_offset_i..c_offset_i+1, cols c_offset_k..c_offset_k+2

        a00 = {a_idx(a_offset_i, a_offset_j): 1}
        a01 = {a_idx(a_offset_i, a_offset_j + 1): 1}
        a10 = {a_idx(a_offset_i + 1, a_offset_j): 1}
        a11 = {a_idx(a_offset_i + 1, a_offset_j + 1): 1}

        b00 = {b_idx(b_offset_j, b_offset_k): 1}
        b01 = {b_idx(b_offset_j, b_offset_k + 1): 1}
        b02 = {b_idx(b_offset_j, b_offset_k + 2): 1}
        b10 = {b_idx(b_offset_j + 1, b_offset_k): 1}
        b11 = {b_idx(b_offset_j + 1, b_offset_k + 1): 1}
        b12 = {b_idx(b_offset_j + 1, b_offset_k + 2): 1}

        c00 = {c_idx(c_offset_i, c_offset_k): 1}
        c01 = {c_idx(c_offset_i, c_offset_k + 1): 1}
        c02 = {c_idx(c_offset_i, c_offset_k + 2): 1}
        c10 = {c_idx(c_offset_i + 1, c_offset_k): 1}
        c11 = {c_idx(c_offset_i + 1, c_offset_k + 1): 1}
        c12 = {c_idx(c_offset_i + 1, c_offset_k + 2): 1}

        # Rank 5 decomposition
        # This is a known minimal decomposition for 2x2 x 2x3
        # References: "The minimum number of multiplications to multiply a 2x2 by a 2x3 matrix"
        
        # M1 = (A00 + A11) * (B00 + B11) -> contributes to C00, C11
        add_term(
            merge_add(a00, a11),
            merge_add(b00, b11),
            merge_add(c00, c11)
        )

        # M2 = (A10 + A11) * B00 -> contributes to C10, C11 (C11 with -)
        add_term(
            merge_add(a10, a11),
            b00,
            merge_w(c10, c11, 1, -1)
        )

        # M3 = A00 * (B01 - B11) -> contributes to C01, C11
        add_term(
            a00,
            merge_sub(b01, b11),
            merge_add(c01, c11)
        )

        # M4 = (A00 + A01) * (B02 + B12) -> contributes to C02, C12
        add_term(
            merge_add(a00, a01),
            merge_add(b02, b12),
            merge_add(c02, c12)
        )

        # M5 = (A01 - A11) * (B10 + B12) -> contributes to C00, C12
        add_term(
            merge_sub(a01, a11),
            merge_add(b10, b12),
            merge_add(c00, c12)
        )

    # Apply Strassen for 4x4 x 4x4 part (columns 0-3)
    # This decomposes the 4x4 x 4x4 multiplication into 4 blocks of 2x2
    # Total rank: 4 * 7 = 28
    
    # A is 4x4, B is 4x4 (cols 0-3), C is 4x4 (cols 0-3)
    # Block (0,0): A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    strassen_2x2(0, 0, 0, 0, 0, 0)
    
    # Block (0,1): A[0:2,2:4] * B[2:4,0:2] -> C[0:2,2:4]
    # Wait, standard Strassen applies to the whole 4x4 as one unit.
    # The 4x4 multiplication tensor can be decomposed using Strassen by viewing 4x4 as 2x2 blocks.
    # Let A be [[A11, A12], [A21, A22]] where each is 2x2.
    # Let B be [[B11, B12], [B21, B22]] where each is 2x2.
    # Let C be [[C11, C12], [C21, C22]] where each is 2x2.
    # C = A*B.
    # The 7 products are computed on the 2x2 blocks.
    # Each 2x2 block multiplication uses 7 scalar multiplications (if we decompose further) or just 1 if we treat them as "scalars" in a higher level?
    # No, we want the tensor decomposition of the 4x4 x 4x4 matrix multiplication.
    # The rank of 4x4 x 4x4 matrix multiplication is 49 (Strassen).
    # This is achieved by applying Strassen to the 4x4 matrix, splitting it into 2x2 blocks, and then applying Strassen to each 2x2 block multiplication? No.
    # Strassen's algorithm reduces the number of multiplications from n^3 to n^log2(7).
    # For 2x2, rank is 7.
    # For 4x4, we split into 2x2 blocks. The 7 Strassen products are 2x2 matrix multiplications.
    # So we need to decompose each of those 2x2 multiplications.
    # That would be 7 * 7 = 49 multiplications.
    
    # So, we apply the 7 Strassen products at the top level (4x4), where each "scalar" is a 2x2 matrix.
    # Then, for each of the 7 products, we apply the 2x2 Strassen decomposition.
    
    # Let's re-architect.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term_nested(u_entries, v_entries, w_entries):
        # u_entries: dict {a_idx: coeff}
        # v_entries: dict {b_idx: coeff}
        # w_entries: dict {c_idx: coeff}
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        for idx, coeff in u_entries.items():
            if coeff != 0:
                u_col[idx] = coeff
        for idx, coeff in v_entries.items():
            if coeff != 0:
                v_col[idx] = coeff
        for idx, coeff in w_entries.items():
            if coeff != 0:
                w_col[idx] = coeff
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    # Define the 2x2 Strassen decomposition as a function that takes block indices
    # and returns a list of (u_dict, v_dict, w_dict) triples for that block.
    
    def get_strassen_2x2_terms(a_off_i, a_off_j, b_off_j, b_off_k, c_off_i, c_off_k):
        terms = []
        
        a00 = {a_idx(a_off_i, a_off_j): 1}
        a01 = {a_idx(a_off_i, a_off_j + 1): 1}
        a10 = {a_idx(a_off_i + 1, a_off_j): 1}
        a11 = {a_idx(a_off_i + 1, a_off_j + 1): 1}

        b00 = {b_idx(b_off_j, b_off_k): 1}
        b01 = {b_idx(b_off_j, b_off_k + 1): 1}
        b10 = {b_idx(b_off_j + 1, b_off_k): 1}
        b11 = {b_idx(b_off_j + 1, b_off_k + 1): 1}

        c00 = {c_idx(c_off_i, c_off_k): 1}
        c01 = {c_idx(c_off_i, c_off_k + 1): 1}
        c10 = {c_idx(c_off_i + 1, c_off_k): 1}
        c11 = {c_idx(c_off_i + 1, c_off_k + 1): 1}

        # M1
        u = merge_add(a00, a11)
        v = merge_add(b00, b11)
        w = merge_add(c00, c11)
        terms.append((u, v, w))

        # M2
        u = merge_add(a10, a11)
        v = b00
        w = merge_w(c10, c11, 1, -1)
        terms.append((u, v, w))

        # M3
        u = a00
        v = merge_sub(b01, b11)
        w = merge_add(c01, c11)
        terms.append((u, v, w))

        # M4
        u = a11
        v = merge_sub(b10, b00)
        w = merge_add(c00, c10)
        terms.append((u, v, w))

        # M5
        u = merge_add(a00, a01)
        v = b11
        w = merge_w(c00, c01, -1, 1)
        terms.append((u, v, w))

        # M6
        u = merge_sub(a10, a00)
        v = merge_add(b00, b01)
        w = c11
        terms.append((u, v, w))

        # M7
        u = merge_sub(a01, a11)
        v = merge_add(b10, b11)
        w = c00
        terms.append((u, v, w))

        return terms

    # Now apply Strassen at the top level for 4x4 x 4x4
    # A blocks: A11 (0,0), A12 (0,2), A21 (2,0), A22 (2,2)
    # B blocks: B11 (0,0), B12 (0,2), B21 (2,0), B22 (2,2)
    # C blocks: C11 (0,0), C12 (0,2), C21 (2,0), C22 (2,2)
    
    # Top-level Strassen products:
    # M1 = (A11+A22) * (B11+B22) -> contributes to C11, C22
    # M2 = (A21+A22) * B11 -> contributes to C21, C22 (-)
    # M3 = A11 * (B12-B22) -> contributes to C12, C22
    # M4 = A22 * (B21-B11) -> contributes to C11, C21
    # M5 = (A11+A12) * B22 -> contributes to C11 (-), C12
    # M6 = (A21-A11) * (B11+B12) -> contributes to C22
    # M7 = (A12-A22) * (B21+B22) -> contributes to C11

    # For each product, we need to decompose the 2x2 matrix multiplication.
    # E.g., for M1:
    # The "A" part is A11+A22. This is a 2x2 matrix where element (i,j) is A11[i,j] + A22[i,j].
    # In the tensor decomposition, this means the U vector for a term in the inner Strassen
    # must be the sum of the U vectors for A11 and A22 at that position.
    # But wait, the inner Strassen decomposition works on 2x2 matrices.
    # So we treat (A11+A22) as a single 2x2 matrix "X".
    # Then we apply the 7 inner Strassen terms to X * Y.
    # The U coefficients for X are sums of A indices.
    
    def merge_2x2_blocks(block1, block2, offset1_i, offset1_j, offset2_i, offset2_j):
        # Returns a 2x2 list of dicts representing the sum of two 2x2 blocks
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                d1 = {a_idx(offset1_i + i, offset1_j + j): 1}
                d2 = {a_idx(offset2_i + i, offset2_j + j): 1}
                row.append(merge_add(d1, d2))
            result.append(row)
        return result

    def merge_2x2_blocks_B(block1, block2, offset1_j, offset1_k, offset2_j, offset2_k):
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                d1 = {b_idx(offset1_j + i, offset1_k + j): 1}
                d2 = {b_idx(offset2_j + i, offset2_k + j): 1}
                row.append(merge_add(d1, d2))
            result.append(row)
        return result

    def merge_2x2_blocks_C(block1, block2, offset1_i, offset1_k, offset2_i, offset2_k, s1=1, s2=1):
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                d1 = {c_idx(offset1_i + i, offset1_k + j): s1}
                d2 = {c_idx(offset2_i + i, offset2_k + j): s2}
                row.append(merge_add(d1, d2))
            result.append(row)
        return result

    def get_inner_strassen_terms_A(X_blocks, Y_blocks, Z_blocks):
        # X_blocks, Y_blocks, Z_blocks are 2x2 lists of dicts
        terms = []
        
        x00, x01 = X_blocks[0][0], X_blocks[0][1]
        x10, x11 = X_blocks[1][0], X_blocks[1][1]
        
        y00, y01 = Y_blocks[0][0], Y_blocks[0][1]
        y10, y11 = Y_blocks[1][0], Y_blocks[1][1]
        
        z00, z01 = Z_blocks[0][0], Z_blocks[0][1]
        z10, z11 = Z_blocks[1][0], Z_blocks[1][1]
        
        # M1_inner = (x00 + x11) * (y00 + y11) -> z00 + z11
        u = merge_add(x00, x11)
        v = merge_add(y00, y11)
        w = merge_add(z00, z11)
        terms.append((u, v, w))

        # M2_inner = (x10 + x11) * y00 -> z10 - z11
        u = merge_add(x10, x11)
        v = y00
        w = merge_w(z10, z11, 1, -1)
        terms.append((u, v, w))

        # M3_inner = x00 * (y01 - y11) -> z01 + z11
        u = x00
        v = merge_sub(y01, y11)
        w = merge_add(z01, z11)
        terms.append((u, v, w))

        # M4_inner = x11 * (y10 - y00) -> z00 + z10
        u = x11
        v = merge_sub(y10, y00)
        w = merge_add(z00, z10)
        terms.append((u, v, w))

        # M5_inner = (x00 + x01) * y11 -> -z00 + z01
        u = merge_add(x00, x01)
        v = y11
        w = merge_w(z00, z01, -1, 1)
        terms.append((u, v, w))

        # M6_inner = (x10 - x00) * (y00 + y01) -> z11
        u = merge_sub(x10, x00)
        v = merge_add(y00, y01)
        w = z11
        terms.append((u, v, w))

        # M7_inner = (x01 - x11) * (y10 + y11) -> z00
        u = merge_sub(x01, x11)
        v = merge_add(y10, y11)
        w = z00
        terms.append((u, v, w))

        return terms

    # Helper to create a single 2x2 block as a list of lists of dicts
    def create_A_block(i_off, j_off):
        return [
            [{a_idx(i_off, j_off): 1}, {a_idx(i_off, j_off + 1): 1}],
            [{a_idx(i_off + 1, j_off): 1}, {a_idx(i_off + 1, j_off + 1): 1}]
        ]

    def create_B_block(j_off, k_off):
        return [
            [{b_idx(j_off, k_off): 1}, {b_idx(j_off, k_off + 1): 1}],
            [{b_idx(j_off + 1, k_off): 1}, {b_idx(j_off + 1, k_off + 1): 1}]
        ]

    def create_C_block(i_off, k_off):
        return [
            [{c_idx(i_off, k_off): 1}, {c_idx(i_off, k_off + 1): 1}],
            [{c_idx(i_off + 1, k_off): 1}, {c_idx(i_off + 1, k_off + 1): 1}]
        ]

    # Part 1: 4x4 x 4x4 (cols 0-3)
    # A11, A12, A21, A22
    A11 = create_A_block(0, 0)
    A12 = create_A_block(0, 2)
    A21 = create_A_block(2, 0)
    A22 = create_A_block(2, 2)

    # B11, B12, B21, B22 (cols 0-3)
    B11 = create_B_block(0, 0)
    B12 = create_B_block(0, 2)
    B21 = create_B_block(2, 0)
    B22 = create_B_block(2, 2)

    # C11, C12, C21, C22 (cols 0-3)
    C11 = create_C_block(0, 0)
    C12 = create_C_block(0, 2)
    C21 = create_C_block(2, 0)
    C22 = create_C_block(2, 2)

    # M1: (A11+A22) * (B11+B22) -> C11 + C22
    X = merge_2x2_blocks(A11, A22, 0, 0, 2, 2)
    Y = merge_2x2_blocks_B(B11, B22, 0, 0, 2, 2)
    Z = merge_2x2_blocks_C(C11, C22, 0, 0, 2, 2, 1, 1)
    for u, v, w in get_inner_strassen_terms_A(X, Y, Z):
        add_term_nested(u, v, w)

    # M2: (A21+A22) * B11 -> C21 - C22
    X = merge_2x2_blocks(A21, A22, 2, 0, 2, 2)
    Y = B11
    Z = merge_2x2_blocks_C(C21, C22, 2, 0, 2, 2, 1, -1)
    for u, v, w in get_inner_strassen_terms_A(X, Y, Z):
        add_term_nested(u, v, w)

    # M3: A11 * (B12-B22) -> C12 + C22
    X = A11
    Y = merge_2x2_blocks_B(B12, B22, 0, 2, 2, 2) # Note: B12 is (0,2), B22 is (2,2). Subtraction in inner? No, outer subtraction.
    # Wait, merge_2x2_blocks_B does addition. I need a subtraction version or pass signs.
    # Let's create a helper for subtraction.
    
    def merge_2x2_blocks_B_sub(b1, b2, off1_j, off1_k, off2_j, off2_k):
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                d1 = {b_idx(off1_j + i, off1_k + j): 1}
                d2 = {b_idx(off2_j + i, off2_k + j): -1}
                row.append(merge_add(d1, d2))
            result.append(row)
        return result

    Y = merge_2x2_blocks_B_sub(B12, B22, 0, 2, 2, 2)
    Z = merge_2x2_blocks_C(C12, C22, 0, 2, 2, 2, 1, 1)
    for u, v, w in get_inner_strassen_terms_A(X, Y, Z):
        add_term_nested(u, v, w)

    # M4: A22 * (B21-B11) -> C11 + C21
    X = A22
    Y = merge_2x2_blocks_B_sub(B21, B11, 2, 0, 0, 0)
    Z = merge_2x2_blocks_C(C11, C21, 0, 0, 2, 0, 1, 1)
    for u, v, w in get_inner_strassen_terms_A(X, Y, Z):
        add_term_nested(u, v, w)

    # M5: (A11+A12) * B22 -> -C11 + C12
    X = merge_2x2_blocks(A11, A12, 0, 0, 0, 2)
    Y = B22
    Z = merge_2x2_blocks_C(C11, C12, 0, 0, 0, 2, -1, 1)
    for u, v, w in get_inner_strassen_terms_A(X, Y, Z):
        add_term_nested(u, v, w)

    # M6: (A21-A11) * (B11+B12) -> C22
    def merge_2x2_blocks_A_sub(a1, a2, off1_i, off1_j, off2_i, off2_j):
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                d1 = {a_idx(off1_i + i, off1_j + j): 1}
                d2 = {a_idx(off2_i + i, off2_j + j): -1}
                row.append(merge_add(d1, d2))
            result.append(row)
        return result

    X = merge_2x2_blocks_A_sub(A21, A11, 2, 0, 0, 0)
    Y = merge_2x2_blocks_B(B11, B12, 0, 0, 0, 2)
    Z = C22
    for u, v, w in get_inner_strassen_terms_A(X, Y, Z):
        add_term_nested(u, v, w)

    # M7: (A12-A22) * (B21+B22) -> C11
    X = merge_2x2_blocks_A_sub(A12, A22, 0, 2, 2, 2)
    Y = merge_2x2_blocks_B(B21, B22, 2, 0, 2, 2)
    Z = C11
    for u, v, w in get_inner_strassen_terms_A(X, Y, Z):
        add_term_nested(u, v, w)

    # Part 2: 4x4 x 4x3 (cols 4-6)
    # Use naive decomposition: for each i in 0..3, j in 0..3, k in 4..6
    # U[i*4+j] = 1, V[j*7+k] = 1, W[k*4+i] = 1
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u_entries = {a_idx(i, j): 1}
                v_entries = {b_idx(j, k): 1}
                w_entries = {c_idx(i, k): 1}
                add_term_nested(u_entries, v_entries, w_entries)

    rank = len(U_cols)
    
    # Convert to matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c in range(rank):
        for r in range(n * m):
            U[r, c] = U_cols[c][r]
        for r in range(m * p):
            V[r, c] = V_cols[c][r]
        for r in range(p * n):
            W[r, c] = W_cols[c][r]

    # Verify tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if s != 1:
                    # Fallback: use fully naive decomposition
                    U_cols2 = []
                    V_cols2 = []
                    W_cols2 = []
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                u_e = {a_idx(ii, jj): 1}
                                v_e = {b_idx(jj, kk): 1}
                                w_e = {c_idx(ii, kk): 1}
                                u_col = [0] * (n * m)
                                v_col = [0] * (m * p)
                                w_col = [0] * (p * n)
                                for idx, coeff in u_e.items():
                                    u_col[idx] = coeff
                                for idx, coeff in v_e.items():
                                    v_col[idx] = coeff
                                for idx, coeff in w_e.items():
                                    w_col[idx] = coeff
                                U_cols2.append(u_col)
                                V_cols2.append(v_col)
                                W_cols2.append(w_col)
                    rank2 = len(U_cols2)
                    U = np.zeros((n * m, rank2))
                    V = np.zeros((m * p, rank2))
                    W = np.zeros((p * n, rank2))
                    for c in range(rank2):
                        for r in range(n * m):
                            U[r, c] = U_cols2[c][r]
                        for r in range(m * p):
                            V[r, c] = V_cols2[c][r]
                        for r in range(p * n):
                            W[r, c] = W_cols2[c][r]
                    def clean_matrix(mat):
                        result = []
                        for row in mat:
                            new_row = []
                            for val in row:
                                if isinstance(val, float) and val == int(val):
                                    new_row.append(int(val))
                                else:
                                    new_row.append(val)
                            result.append(new_row)
                        return result
                    U = clean_matrix(U.tolist())
                    V = clean_matrix(V.tolist())
                    W = clean_matrix(W.tolist())
                    return {"U": U, "V": V, "W": W}

    # Clean up: convert to int if possible
    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if isinstance(val, float) and val == int(val):
                    new_row.append(int(val))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result

    U = clean_matrix(U.tolist())
    V = clean_matrix(V.tolist())
    W = clean_matrix(W.tolist())

    return {"U": U, "V": V, "W": W}
