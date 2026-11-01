"""Matrix multiplication 2x2 x 2x8 via Strassen-style decomposition (rank 28).

Approach:
  - Split the 2x8 matrix into four 2x2 blocks B0,B1,B2,B3.
  - Compute C_k = A * B_k for k=0..3 using Strassen's 7-multiplication
    scheme applied independently to each block.
  - The 28 scalar multiplications give a rank-28 bilinear decomposition.
  - Each Strassen term is a rank-1 tensor in the (A,B,C) space.
"""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=8

    # Strassen's 7 terms for 2x2 matrix multiplication:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b12
    # M3 = a11*(b11-b21)
    # M4 = a22*(b22-b12)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b21)
    # M7 = (a12-a22)*(b12+b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For each 2x2 block k (k=0,1,2,3) of the 2x8 matrix:
    #   B_k = [[b_k0, b_k1], [b_k2, b_k3]]  (rows j=0,1; cols offset 4k)
    #   The indices in the V matrix: row = j*p + col, so for block k:
    #     b_k0 -> V[0*8 + 4k], b_k1 -> V[0*8 + 4k+1]
    #     b_k2 -> V[1*8 + 4k], b_k3 -> V[1*8 + 4k+3]
    #
    # U indices: i*m + j where i is row of A, j is col of A (i.e., row of B)
    #   a11 -> U[0*2+0] = U[0]
    #   a12 -> U[0*2+1] = U[1]
    #   a21 -> U[1*2+0] = U[2]
    #   a22 -> U[1*2+1] = U[3]
    #
    # W indices: k*n + i where k is output row, i is output col
    #   For block k, output rows are at positions 4k, 4k+1 (cols in C)
    #   W index = col_in_C * n + row_in_C
    #   C11 of block k -> W[(4k)*2 + 0] = W[8k]
    #   C12 of block k -> W[(4k+1)*2 + 0] = W[8k+2]
    #   C21 of block k -> W[(4k)*2 + 1] = W[8k+1]
    #   C22 of block k -> W[(4k+1)*2 + 1] = W[8k+3]

    rank = 28  # 7 Strassen terms * 4 blocks

    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28

    # Strassen term definitions:
    # Each term: (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    # u: which A elements (indices 0-3), v: which B elements (local 0-3), w: which C elements (local 0-3)

    strassen_terms = [
        # M1 = (a11+a22)*(b00+b11) -> contributes to C00 (+1), C11 (+1)
        (([0, 3], [1, 1]), ([0, 3], [1, 1]), ([0, 3], [1, 1])),
        # M2 = (a21+a22)*b01 -> contributes to C10 (+1), C11 (-1)
        (([2, 3], [1, 1]), ([1], [1]), ([1, 3], [1, -1])),
        # M3 = a11*(b00-b10) -> contributes to C01 (+1), C11 (+1)
        (([0], [1]), ([0, 2], [1, -1]), ([1, 3], [1, 1])),
        # M4 = a22*(b11-b01) -> contributes to C00 (+1), C10 (+1)
        (([3], [1]), ([3, 1], [1, -1]), ([0, 1], [1, 1])),
        # M5 = (a11+a12)*b11 -> contributes to C00 (-1), C01 (+1)
        (([0, 1], [1, 1]), ([3], [1]), ([0, 1], [-1, 1])),
        # M6 = (a21-a11)*(b00+b10) -> contributes to C11 (+1)
        (([2, 0], [1, -1]), ([0, 2], [1, 1]), ([3], [1])),
        # M7 = (a12-a22)*(b01+b11) -> contributes to C00 (+1), C11 (-1)
        (([1, 3], [1, -1]), ([1, 3], [1, 1]), ([0, 3], [1, -1])),
    ]

    for block in range(4):
        for t in range(7):
            col = block * 7 + t
            u_idx, u_coef = strassen_terms[t][0]
            v_idx, v_coef = strassen_terms[t][1]
            w_idx, w_coef = strassen_terms[t][2]

            # U: A indices are 0,1,2,3 (a11,a12,a21,a22)
            for ui, uc in zip(u_idx, u_coef):
                U[ui][col] += uc

            # V: B local indices 0,1,2,3 map to global row = j*8 + (4*block + local_col)
            #   local 0 -> (row=0, col=4*block)
            #   local 1 -> (row=0, col=4*block+1)
            #   local 2 -> (row=1, col=4*block)
            #   local 3 -> (row=1, col=4*block+1)
            for vi, vc in zip(v_idx, v_coef):
                local_row = vi // 2
                local_col = vi % 2
                global_row = local_row * p + 4 * block + local_col
                V[global_row][col] += vc

            # W: C local indices 0,1,2,3 (C00,C01,C10,C11) map to global
            #   local 0 -> (col=4*block, row=0) -> W[0*n + 4*block] = W[4*block]
            #   local 1 -> (col=4*block+1, row=0) -> W[4*block+1]
            #   local 2 -> (col=4*block, row=1) -> W[1*n + 4*block] = W[2 + 4*block]
            #   local 3 -> (col=4*block+1, row=1) -> W[3 + 4*block]
            for wi, wc in zip(w_idx, w_coef):
                local_row = wi // 2
                local_col = wi % 2
                global_col = 4 * block + local_col
                global_row = local_row
                W[global_row * n + global_col][col] += wc

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
