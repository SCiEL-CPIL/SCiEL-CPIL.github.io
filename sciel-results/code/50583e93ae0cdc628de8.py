"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-98 exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    Strategy: Use Strassen's algorithm for 2x2x2 matrix multiplication.
    - First, decompose the 4x4 * 4x8 problem into four 2x2 * 2x4 sub-problems.
    - For each 2x2 * 2x4 sub-problem, further decompose into two 2x2 * 2x2 problems
      (one for the left 2 columns of B, one for the right 2 columns).
    - Each 2x2 * 2x2 multiplication uses Strassen's 7-term algorithm.
    - Total: 4 (2x2 blocks of A) * 2 (2x2 blocks of B) * 7 (Strassen terms) = 56 terms?
    
    Wait, let me reconsider. The problem is 4x4 * 4x8 = 4x8.
    
    Let's use a direct Strassen-like approach for the full 4x4 * 4x8:
    
    Actually, let me use the standard approach:
    - Partition A (4x4) into four 2x2 blocks: A11, A12, A21, A22
    - Partition B (4x8) into four 2x4 blocks: B11, B12, B21, B22 (rows 1-2 vs 3-4)
      Or better: partition B into eight 2x2 blocks: B11, B12, B13, B14, B21, B22, B23, B24
    - Result C (4x8) has eight 2x2 blocks.
    
    For each pair (Ai, Bj) where i is row-block of A and j is col-block of B,
    we compute a 2x2 * 2x2 product using Strassen (7 multiplications).
    
    Number of such pairs: 2 (row blocks of A) * 4 (col blocks of B) = 8 pairs.
    Each requires 7 Strassen multiplications.
    Total: 8 * 7 = 56 multiplications.
    
    But we can share some of the Strassen intermediate terms!
    
    Actually, let me just implement the straightforward Strassen for each 2x2*2x2 block
    without sharing, giving rank 56. But wait, that might not be correct because
    the Strassen terms for different B blocks are independent.
    
    Let me just code it up carefully.
    
    A is 4x4, indexed A[i,j] for i=0..3, j=0..3.
    B is 4x8, indexed B[j,k] for j=0..3, k=0..7.
    C is 4x8, indexed C[i,k] for i=0..3, k=0..7.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Partition:
    - A row blocks: rows 0-1 (block 0), rows 2-3 (block 1)
    - B col blocks: cols 0-1 (block 0), cols 2-3 (block 1), cols 4-5 (block 2), cols 6-7 (block 3)
    - B row blocks: rows 0-1 (block 0), rows 2-3 (block 1)
    
    For each (a_row_block, b_row_block, b_col_block), we compute a 2x2 * 2x2 product.
    a_row_block in {0,1}, b_row_block in {0,1}, b_col_block in {0,1,2,3}.
    
    That's 2*2*4 = 16 such 2x2*2x2 products, each with 7 Strassen terms = 112 terms.
    But that's more than schoolbook (128)? No, 112 < 128, good.
    
    Wait, I'm overcomplicating. Let me use the standard Strassen recursion:
    
    For 4x4 * 4x4, Strassen gives 4^log2(7) = 49 multiplications.
    For 4x4 * 4x8, we can think of it as two 4x4 * 4x4 problems, so 98 multiplications.
    
    Let me implement Strassen for 4x4 * 4x4, then double it for the 4x8 case.
    
    Strassen 4x4 * 4x4:
    Partition A into four 2x2 blocks: A00, A01, A10, A11
    Partition B into four 2x2 blocks: B00, B01, B10, B11
    Compute 7 intermediate 2x2*2x2 products using Strassen:
      M1 = (A00 + A11) * (B00 + B11)
      M2 = (A10 + A11) * B00
      M3 = A00 * (B01 - B11)
      M4 = A11 * (B10 - B00)
      M5 = (A00 + A01) * B11
      M6 = (A10 - A00) * (B00 + B01)
      M7 = (A01 - A11) * (B10 + B11)
    
    Each Mi is a 2x2 matrix computed via 7 scalar multiplications.
    Total: 7 * 7 = 49 scalar multiplications.
    
    For 4x4 * 4x8:
    Split B into B_left (4x4, cols 0-3) and B_right (4x4, cols 4-7).
    C_left = A * B_left (4x4)
    C_right = A * B_right (4x4)
    Total: 49 + 49 = 98 scalar multiplications.
    
    Now I need to construct U, V, W for this decomposition.
    """
    import time
    start_time = time.time()
    DEADLINE = 145.0  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We'll build the Strassen decomposition for 4x4 * 4x4, then use it twice.
    # First, define the Strassen terms for 2x2 * 2x2.
    # Each term is (u, v, w) where u is a 4-vector (coefficients for A's 2x2 block),
    # v is a 4-vector (coefficients for B's 2x2 block), and w is a 4x4 matrix of
    # contributions to the output 2x2 block (flattened to 4 entries).
    
    # Strassen 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M3
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6 + M7
    
    # For each Strassen term, we have:
    # - u: coefficients for A (4 entries: a,b,c,d)
    # - v: coefficients for B (4 entries: e,f,g,h)
    # - w: 4x4 matrix showing how this scalar product contributes to C
    
    # Let's define the 7 Strassen terms for 2x2*2x2:
    strassen_2x2 = []
    
    # M1 = (a+d)(e+h)
    u1 = [1, 0, 0, 1]  # a + d
    v1 = [1, 0, 0, 1]  # e + h
    # Contribution to C:
    # C11 gets +1, C12 gets +1, C21 gets -1, C22 gets +1
    w1 = [[1, 1], [-1, 1]]
    strassen_2x2.append((u1, v1, w1))
    
    # M2 = (c+d)e
    u2 = [0, 0, 1, 1]  # c + d
    v2 = [1, 0, 0, 0]  # e
    w2 = [[0, 0], [1, 0]]  # C21 += M2
    strassen_2x2.append((u2, v2, w2))
    
    # M3 = a(f-h)
    u3 = [1, 0, 0, 0]  # a
    v3 = [0, 1, 0, -1]  # f - h
    w3 = [[1, 1], [0, 0]]  # C11 += M3, C12 += M3
    strassen_2x2.append((u3, v3, w3))
    
    # M4 = d(g-e)
    u4 = [0, 0, 0, 1]  # d
    v4 = [0, 0, 1, 0]  # g - e (wait, e is index 0, g is index 2)
    v4 = [-1, 0, 1, 0]  # g - e
    w4 = [[1, 0], [1, 0]]  # C11 += M4, C21 += M4
    strassen_2x2.append((u4, v4, w4))
    
    # M5 = (a+b)h
    u5 = [1, 1, 0, 0]  # a + b
    v5 = [0, 0, 0, 1]  # h
    w5 = [[-1, 1], [0, 0]]  # C11 -= M5, C12 += M5
    strassen_2x2.append((u5, v5, w5))
    
    # M6 = (c-a)(e+f)
    u6 = [0, 0, 1, -1]  # c - a (wait, a is index 0, c is index 2)
    u6 = [-1, 0, 1, 0]  # c - a
    v6 = [1, 1, 0, 0]  # e + f
    w6 = [[0, 0], [0, 1]]  # C22 += M6
    strassen_2x2.append((u6, v6, w6))
    
    # M7 = (b-d)(g+h)
    u7 = [0, 1, 0, -1]  # b - d
    v7 = [0, 0, 1, 1]  # g + h
    w7 = [[0, 0], [0, 1]]  # C22 += M7
    strassen_2x2.append((u7, v7, w7))
    
    # Now, for 4x4 * 4x4, we use Strassen at the 2x2 block level.
    # A is partitioned into A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3),
    # A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B is partitioned into B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3),
    # B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    
    # The 7 Strassen terms at the 4x4 level are:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Each Mi is a 2x2 matrix, computed via 7 scalar multiplications.
    # We need to express the coefficients for A and B for each scalar multiplication.
    
    # For A, we have 16 entries (4x4). For each Strassen term at the 4x4 level,
    # the A coefficient vector is a 16-vector.
    
    # Let's define helper functions to build 16-vectors from 2x2 block specifications.
    
    def a_block_vector(block_rows, block_cols, coeffs):
        """Build a 16-vector (for 4x4 A) with given coefficients for a 2x2 block."""
        vec = [0] * 16
        for i in range(2):
            for j in range(2):
                row = block_rows[0] + i
                col = block_cols[0] + j
                vec[row * 4 + col] = coeffs[i * 2 + j]
        return vec
    
    def b_block_vector(block_rows, block_cols, coeffs):
        """Build a 16-vector (for 4x4 B) with given coefficients for a 2x2 block."""
        vec = [0] * 16
        for i in range(2):
            for j in range(2):
                row = block_rows[0] + i
                col = block_cols[0] + j
                vec[row * 4 + col] = coeffs[i * 2 + j]
        return vec
    
    # Block positions:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    A00_pos = ((0, 1), (0, 1))
    A01_pos = ((0, 1), (2, 3))
    A10_pos = ((2, 3), (0, 1))
    A11_pos = ((2, 3), (2, 3))
    
    B00_pos = ((0, 1), (0, 1))
    B01_pos = ((0, 1), (2, 3))
    B10_pos = ((2, 3), (0, 1))
    B11_pos = ((2, 3), (2, 3))
    
    # For each of the 7 Strassen terms at the 4x4 level, we have a combination of
    # A blocks and B blocks. For each such combination, we then apply the 7 Strassen
    # terms at the 2x2 level.
    
    # The 4x4-level Strassen terms:
    four_x_four_strassen = [
        # (A_blocks, B_blocks, output_block)
        # M1 = (A00 + A11) * (B00 + B11) -> contributes to C00, C01, C10, C11
        ('A00+A11', 'B00+B11', 'C00+C01+C10+C11'),
        # M2 = (A10 + A11) * B00 -> contributes to C10
        ('A10+A11', 'B00', 'C10'),
        # M3 = A00 * (B01 - B11) -> contributes to C01
        ('A00', 'B01-B11', 'C01'),
        # M4 = A11 * (B10 - B00) -> contributes to C11
        ('A11', 'B10-B00', 'C11'),
        # M5 = (A00 + A01) * B11 -> contributes to C01
        ('A00+A01', 'B11', 'C01'),
        # M6 = (A10 - A00) * (B00 + B01) -> contributes to C11
        ('A10-A00', 'B00+B01', 'C11'),
        # M7 = (A01 - A11) * (B10 + B11) -> contributes to C11
        ('A01-A11', 'B10+B11', 'C11'),
    ]
    
    # This is getting complex. Let me just directly construct the U, V, W matrices
    # by enumerating all the scalar multiplications.
    
    # For 4x4 * 4x4:
    # We'll create a list of (u_vec, v_vec, w_matrix) where:
    # u_vec is a 16-vector (coefficients for A)
    # v_vec is a 16-vector (coefficients for B)
    # w_matrix is a 16x16 matrix (contribution to C, flattened)
    
    # Actually, let me just hardcode the 49 terms for 4x4*4x4 using the known
    # Strassen construction, then duplicate for the second half of B.
    
    # For now, let me use a simpler approach: just use schoolbook but with
    # some obvious factorizations.
    
    # Actually, the simplest correct approach with rank < 128 is to use
    # Strassen for 2x2*2x2 and apply it 8 times (for the 4x4*4x8 case,
    # we have 2*4=8 such 2x2*2x2 products if we partition appropriately).
    
    # Let me just implement the 56-term decomposition:
    # Partition A into 2 row-blocks (each 2x4), B into 2 row-blocks * 4 col-blocks
    # (each 2x2), giving 2*2*4 = 16 2x2*2x2 products, each with 7 terms = 112.
    
    # Hmm, that's 112, which is less than 128 but not optimal.
    
    # Let me just use the 98-term Strassen approach and implement it correctly.
    
    # I'll construct the decomposition step by step.
    
    # For 4x4 * 4x4, the Strassen decomposition has 49 terms.
    # For 4x4 * 4x8, we have two independent 4x4 * 4x4 problems, so 98 terms.
    
    # Let me define the 49 terms for 4x4 * 4x4 explicitly.
    
    # Each term is (u, v, w) where:
    # u: 16-vector for A (4x4)
    # v: 16-vector for B (4x4)
    # w: 16x16 matrix for C (4x4) - but actually w is a 4x4 matrix, flattened
    
    terms_4x4 = []
    
    # Define the Strassen terms at the 4x4 level
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # For each Mi, we need to express the 2x2*2x2 product using 7 scalar multiplications.
    
    # Let me define a function to combine 2x2 block operations into 16-vectors.
    
    def combine_a_blocks(blocks_with_coeffs):
        """
        Combine multiple 2x2 A blocks with coefficients into a 16-vector.
        blocks_with_coeffs: list of (block_name, coeffs_4_vector)
        """
        vec = [0] * 16
        block_positions = {
            'A00': (0, 0),  # (start_row, start_col)
            'A01': (0, 2),
            'A10': (2, 0),
            'A11': (2, 2),
        }
        for block_name, coeffs in blocks_with_coeffs:
            sr, sc = block_positions[block_name]
            for i in range(2):
                for j in range(2):
                    vec[(sr + i) * 4 + (sc + j)] += coeffs[i * 2 + j]
        return vec
    
    def combine_b_blocks(blocks_with_coeffs):
        """
        Combine multiple 2x2 B blocks with coefficients into a 16-vector.
        blocks_with_coeffs: list of (block_name, coeffs_4_vector)
        """
        vec = [0] * 16
        block_positions = {
            'B00': (0, 0),
            'B01': (0, 2),
            'B10': (2, 0),
            'B11': (2, 2),
        }
        for block_name, coeffs in blocks_with_coeffs:
            sr, sc = block_positions[block_name]
            for i in range(2):
                for j in range(2):
                    vec[(sr + i) * 4 + (sc + j)] += coeffs[i * 2 + j]
        return vec
    
    def combine_c_blocks(blocks_with_matrices):
        """
        Combine multiple 2x2 C blocks with 2x2 matrices into a 16x16 matrix (flattened to 256-vector? No, 16x16).
        blocks_with_matrices: list of (block_name, 2x2_matrix)
        Returns a 4x4 matrix.
        """
        mat = [[0] * 4 for _ in range(4)]
        block_positions = {
            'C00': (0, 0),
            'C01': (0, 2),
            'C10': (2, 0),
            'C11': (2, 2),
        }
        for block_name, m2 in blocks_with_matrices:
            sr, sc = block_positions[block_name]
            for i in range(2):
                for j in range(2):
                    mat[sr + i][sc + j] += m2[i][j]
        return mat
    
    # Now define the 7 Strassen terms for 2x2*2x2 as before:
    s2x2 = strassen_2x2
    
    # For M1 = (A00 + A11) * (B00 + B11):
    # A part: A00 with coeff [1,0,0,1], A11 with coeff [1,0,0,1]
    # B part: B00 with coeff [1,0,0,1], B11 with coeff [1,0,0,1]
    # Output: C00, C01, C10, C11 all get the same 2x2 pattern from w1
    
    a_m1 = combine_a_blocks([('A00', [1, 0, 0, 1]), ('A11', [1, 0, 0, 1])])
    b_m1 = combine_b_blocks([('B00', [1, 0, 0, 1]), ('B11', [1, 0, 0, 1])])
    for (u_local, v_local, w_local) in s2x2:
        u = combine_a_blocks([('A00', u_local), ('A11', u_local)])
        v = combine_b_blocks([('B00', v_local), ('B11', v_local)])
        w = combine_c_blocks([('C00', w_local), ('C01', w_local), ('C10', w_local), ('C11', w_local)])
        terms_4x4.append((u, v, w))
    
    # M2 = (A10 + A11) * B00
    for (u_local, v_local, w_local) in s2x2:
        u = combine_a_blocks([('A10', u_local), ('A11', u_local)])
        v = combine_b_blocks([('B00', v_local)])
        w = combine_c_blocks([('C10', w_local)])
        terms_4x4.append((u, v, w))
    
    # M3 = A00 * (B01 - B11)
    for (u_local, v_local, w_local) in s2x2:
        u = combine_a_blocks([('A00', u_local)])
        v = combine_b_blocks([('B01', v_local), ('B11', [-x for x in v_local])])
        w = combine_c_blocks([('C01', w_local)])
        terms_4x4.append((u, v, w))
    
    # M4 = A11 * (B10 - B00)
    for (u_local, v_local, w_local) in s2x2:
        u = combine_a_blocks([('A11', u_local)])
        v = combine_b_blocks([('B10', v_local), ('B00', [-x for x in v_local])])
        w = combine_c_blocks([('C11', w_local)])
        terms_4x4.append((u, v, w))
    
    # M5 = (A00 + A01) * B11
    for (u_local, v_local, w_local) in s2x2:
        u = combine_a_blocks([('A00', u_local), ('A01', u_local)])
        v = combine_b_blocks([('B11', v_local)])
        w = combine_c_blocks([('C01', w_local)])
        terms_4x4.append((u, v, w))
    
    # M6 = (A10 - A00) * (B00 + B01)
    for (u_local, v_local, w_local) in s2x2:
        u = combine_a_blocks([('A10', u_local), ('A00', [-x for x in u_local])])
        v = combine_b_blocks([('B00', v_local), ('B01', v_local)])
        w = combine_c_blocks([('C11', w_local)])
        terms_4x4.append((u, v, w))
    
    # M7 = (A01 - A11) * (B10 + B11)
    for (u_local, v_local, w_local) in s2x2:
        u = combine_a_blocks([('A01', u_local), ('A11', [-x for x in u_local])])
        v = combine_b_blocks([('B10', v_local), ('B11', v_local)])
        w = combine_c_blocks([('C11', w_local)])
        terms_4x4.append((u, v, w))
    
    # Now we have 49 terms for 4x4 * 4x4.
    # For 4x4 * 4x8, we duplicate these 49 terms for the second half of B (cols 4-7).
    
    # But wait, the B matrix for the second half has a different structure.
    # B is 4x8, so B_left is cols 0-3, B_right is cols 4-7.
    
    # Let me construct the full U, V, W for 4x4 * 4x8.
    # U: 16 x rank
    # V: 32 x rank (4 rows * 8 cols)
    # W: 32 x rank (8 cols * 4 rows, following W[k*n+i, r] convention)
    
    # The trace convention: W[k*n+i, r] means W is indexed by (k, i) where
    # k is the column index of C (0-7) and i is the row index of C (0-3).
    # So W has shape (8*4, rank) = (32, rank).
    
    rank = len(terms_4x4) * 2  # 98
    
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    for idx, (u, v, w) in enumerate(terms_4x4):
        # First copy: B cols 0-3
        col = idx
        for i in range(16):
            U[i][col] = u[i]
        for j in range(16):
            # v[j] corresponds to B[row, col] where row = j//4, col = j%4
            row = j // 4
            col_b = j % 4
            V[row * 8 + col_b][col] = v[j]
        # w is a 4x4 matrix for C rows 0-3, cols 0-3
        for i in range(4):
            for k in range(4):
                W[k * 4 + i][col] = w[i][k]
        
        # Second copy: B cols 4-7
        col2 = idx + 49
        for i in range(16):
            U[i][col2] = u[i]
        for j in range(16):
            row = j // 4
            col_b = j % 4
            V[row * 8 + (col_b + 4)][col2] = v[j]
        for i in range(4):
            for k in range(4):
                W[(k + 4) * 4 + i][col2] = w[i][k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
