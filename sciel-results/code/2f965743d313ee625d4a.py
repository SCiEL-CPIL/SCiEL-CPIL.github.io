"""Strassen-based decomposition for 2x4 * 4x4 matrix multiplication.

This program constructs a rank-28 exact tensor decomposition of the
2x4-by-4x4 matrix multiplication tensor by treating it as four independent
2x2-by-2x2 block multiplications and applying Strassen's algorithm to each.
"""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4

    # Strassen's 7 multiplication formulas for a 2x2 matrix product C = A*B:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Each block is a 2x2 submatrix multiplication.
    # Block (bi, bj) corresponds to:
    #   A rows [2*bi, 2*bi+1], cols [2*bj, 2*bj+1]
    #   B rows [2*bj, 2*bj+1], cols [2*bi, 2*bi+1]  -- wait, let me think carefully.
    #
    # Actually for 2x4 * 4x4:
    #   A is 2x4, B is 4x4, C is 2x4.
    #   Split into 2x2 blocks:
    #     A = [A11 A12] where A11 is 2x2 (rows 0-1, cols 0-1), A12 is 2x2 (rows 0-1, cols 2-3)
    #     B = [B11 B12; B21 B22] where each is 2x2
    #     C = [C11 C12] where each is 2x2
    #   C11 = A11*B11 + A12*B21
    #   C12 = A11*B12 + A12*B22
    #
    # So we have 4 independent 2x2*2x2 multiplications:
    #   Product 1: A11 * B11 -> contributes to C11
    #   Product 2: A12 * B21 -> contributes to C11
    #   Product 3: A11 * B12 -> contributes to C12
    #   Product 4: A12 * B22 -> contributes to C12
    #
    # Each 2x2*2x2 uses 7 Strassen terms, so total rank = 4*7 = 28.

    rank = 28

    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 16 rows
    W = [[0] * rank for _ in range(p * n)]  # 8 rows

    # Define the 4 block multiplications
    # Each block: (A_block_start_i, A_block_start_j, B_block_start_j, B_block_start_k, C_block_start_k)
    # where A is indexed by (i, j) with i in [0,n), j in [0,m)
    # B is indexed by (j, k) with j in [0,m), k in [0,p)
    # C is indexed by (k, i) with k in [0,p), i in [0,n)
    #
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2]
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[2:4, 0:2]

    # Strassen formulas: for each term t in 0..6, define:
    #   a_indices: list of (row, col) in the 2x2 A block with coefficients
    #   b_indices: list of (row, col) in the 2x2 B block with coefficients
    #   c_indices: list of (row, col) in the 2x2 C block with coefficients (for output)

    # Strassen terms for C = A*B where A, B, C are 2x2:
    # M1: (A[0][0]+A[1][1]) * (B[0][0]+B[1][1])
    # M2: (A[1][0]+A[1][1]) * B[0][0]
    # M3: A[0][0] * (B[0][1]-B[1][1])
    # M4: A[1][1] * (B[1][0]-B[0][0])
    # M5: (A[0][0]+A[0][1]) * B[1][1]
    # M6: (A[1][0]-A[0][0]) * (B[0][0]+B[0][1])
    # M7: (A[0][1]-A[1][1]) * (B[1][0]+B[1][1])
    #
    # C[0][0] = M1 + M4 - M5 + M7
    # C[0][1] = M3 + M5
    # C[1][0] = M2 + M4
    # C[1][1] = M1 - M2 + M3 + M6

    # For each Strassen term, define the linear combinations for A, B
    # and which C elements it contributes to with what sign.
    #
    # We'll represent each term as:
    #   a_part: list of ((local_i, local_j), coeff)
    #   b_part: list of ((local_j, local_k), coeff)
    #   c_contrib: list of ((local_k, local_i), coeff)

    strassen_terms = [
        # M1
        {
            'a': [((0,0),1), ((1,1),1)],
            'b': [((0,0),1), ((1,1),1)],
            'c': [((0,0),1), ((1,1),1)]
        },
        # M2
        {
            'a': [((1,0),1), ((1,1),1)],
            'b': [((0,0),1)],
            'c': [((1,0),1), ((1,1),-1)]
        },
        # M3
        {
            'a': [((0,0),1)],
            'b': [((0,1),1), ((1,1),-1)],
            'c': [((0,1),1), ((1,1),1)]
        },
        # M4
        {
            'a': [((1,1),1)],
            'b': [((1,0),1), ((0,0),-1)],
            'c': [((0,0),1), ((1,0),1)]
        },
        # M5
        {
            'a': [((0,0),1), ((0,1),1)],
            'b': [((1,1),1)],
            'c': [((0,0),-1), ((0,1),1)]
        },
        # M6
        {
            'a': [((1,0),1), ((0,0),-1)],
            'b': [((0,0),1), ((0,1),1)],
            'c': [((1,1),1)]
        },
        # M7
        {
            'a': [((0,1),1), ((1,1),-1)],
            'b': [((1,0),1), ((1,1),1)],
            'c': [((0,0),1)]
        },
    ]

    # 4 blocks with their starting indices
    # Block: (a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start)
    # c_row is k-index, c_col is i-index in the W convention W[k*n+i]
    blocks = [
        # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (k=0,1; i=0,1)
        (0, 0, 0, 0, 0, 0),
        # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        (0, 2, 2, 0, 0, 0),
        # A[0:2, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2] (k=2,3; i=0,1)
        (0, 0, 0, 2, 2, 0),
        # A[0:2, 2:4] * B[2:4, 2:4] -> C[2:4, 0:2]
        (0, 2, 2, 2, 2, 0),
    ]

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(blocks):
        for term_idx, term in enumerate(strassen_terms):
            col = block_idx * 7 + term_idx

            # Fill U
            for (li, lj), coeff in term['a']:
                u_row = (ar + li) * m + (ac + lj)
                U[u_row][col] += coeff

            # Fill V
            for (lj, lk), coeff in term['b']:
                v_row = (br + lj) * p + (bc + lk)
                V[v_row][col] += coeff

            # Fill W
            for (lk, li), coeff in term['c']:
                w_row = (cr + lk) * n + (cc + li)
                W[w_row][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
