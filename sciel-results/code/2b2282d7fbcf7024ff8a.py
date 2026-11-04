"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6

    # We use the Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    # For 4x4x6, we split into 2x2 blocks:
    # A is 4x4 -> 2x2 blocks of 2x2
    # B is 4x6 -> 2x3 blocks of 2x2
    # C is 4x6 -> 2x3 blocks of 2x2
    #
    # C[i_block, k_block] = sum_{j_block=0}^{1} A[i_block, j_block] * B[j_block, k_block]
    #
    # For each (i_block, k_block), we have a sum of two 2x2 matrix products.
    # We can use Strassen for each product, giving 7+7=14 terms per (i_block, k_block).
    # Total: 2 * 3 * 14 = 84 terms.
    #
    # But we can do better by combining the two Strassen decompositions.
    # Actually, let's just use the simple approach: for each (i_block, k_block, j_block),
    # use a 2x2x2 Strassen decomposition. That gives 2*3*2*7 = 84 terms.
    #
    # Wait, that's the same. Let me think again.
    #
    # Actually, the cleanest approach: for each of the 6 block positions (i_block, k_block),
    # we have C[i_block, k_block] = A[i_block, 0] * B[0, k_block] + A[i_block, 1] * B[1, k_block]
    # Each 2x2 product has rank 7 via Strassen, so the sum has rank at most 14.
    # Total rank = 6 * 14 = 84.
    #
    # But we can potentially share some structure. For now, let's implement the 84-rank version.
    #
    # Strassen for 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b00)
    # M4 = a11(b01-b11)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    #
    # U vectors (for A, 4 elements: a00, a01, a10, a11):
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, 1]
    # M3: [1, 0, 0, 0]
    # M4: [0, 0, 0, 1]
    # M5: [1, 1, 0, 0]
    # M6: [1, 0, -1, 0]
    # M7: [0, 1, 0, -1]
    #
    # V vectors (for B, 4 elements: b00, b01, b10, b11):
    # M1: [1, 0, 0, 1]
    # M2: [1, 0, 0, 0]
    # M3: [0, 0, 1, -1]
    # M4: [0, 1, 0, -1]
    # M5: [0, 0, 0, 1]
    # M6: [1, 1, 0, 0]
    # M7: [0, 0, 1, 1]
    #
    # W vectors (for C, 4 elements in trace convention: C00, C10, C01, C11 -> but wait,
    # the trace convention for the full output is W[k*n+i, r] where the output index is k*n + i.
    # For a 2x2 block at position (row_block, col_block), the local indices are i in {0,1}, k in {0,1}.
    # The global output index is (row_block*2 + i) * (num_col_blocks*2) + (col_block*2 + k)? 
    # No, the output C has shape n x p = 4 x 6.
    # The tensor index for C is (i, k) where i is row, k is column.
    # The trace convention says W[k*n + i, r], so the index is k*n + i.
    # For a 2x2 block at (row_block, col_block), the local (i, k) maps to global (row_block*2+i, col_block*2+k).
    # So the global index is (col_block*2+k)*n + (row_block*2+i) = (col_block*2+k)*4 + (row_block*2+i).
    #
    # Let me define the W vectors for each Strassen term, for a 2x2 block at (row_block, col_block):
    # The 4 local output positions are:
    # (i=0, k=0) -> global index = col_block*2*4 + row_block*2 + 0 = col_block*8 + row_block*2
    # (i=1, k=0) -> global index = col_block*8 + row_block*2 + 1
    # (i=0, k=1) -> global index = (col_block*2+1)*4 + row_block*2 + 0 = col_block*8 + 4 + row_block*2
    # (i=1, k=1) -> global index = col_block*8 + 4 + row_block*2 + 1
    #
    # So for a 2x2 block, the W vector has non-zero entries at these 4 positions.
    #
    # For Strassen term M1: contributes to C00 and C11
    # C00: local (i=0, k=0), coefficient +1
    # C11: local (i=1, k=1), coefficient +1
    # So W1 has +1 at col_block*8 + row_block*2 and +1 at col_block*8 + 4 + row_block*2 + 1
    #
    # For M2: contributes to C10 and C11
    # C10: local (i=1, k=0), coefficient +1
    # C11: local (i=1, k=1), coefficient -1
    # W2: +1 at col_block*8 + row_block*2 + 1, -1 at col_block*8 + 4 + row_block*2 + 1
    #
    # For M3: contributes to C01 and C11
    # C01: local (i=0, k=1), coefficient +1
    # C11: local (i=1, k=1), coefficient +1
    # W3: +1 at col_block*8 + 4 + row_block*2, +1 at col_block*8 + 4 + row_block*2 + 1
    #
    # For M4: contributes to C00
    # C00: local (i=0, k=0), coefficient +1
    # W4: +1 at col_block*8 + row_block*2
    #
    # For M5: contributes to C00 and C01
    # C00: local (i=0, k=0), coefficient -1
    # C01: local (i=0, k=1), coefficient +1
    # W5: -1 at col_block*8 + row_block*2, +1 at col_block*8 + 4 + row_block*2
    #
    # For M6: contributes to C10 and C11
    # C10: local (i=1, k=0), coefficient +1
    # C11: local (i=1, k=1), coefficient +1
    # W6: +1 at col_block*8 + row_block*2 + 1, +1 at col_block*8 + 4 + row_block*2 + 1
    #
    # For M7: contributes to C00
    # C00: local (i=0, k=0), coefficient +1
    # W7: +1 at col_block*8 + row_block*2

    U_strassen = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [1, 0, -1, 0], # M6
        [0, 1, 0, -1], # M7
    ]
    V_strassen = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 1, -1], # M3
        [0, 1, 0, -1], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1],  # M7
    ]
    W_strassen_local = [
        [(0, 1), (3, 1)],      # M1: C00 +1, C11 +1
        [(1, 1), (3, -1)],     # M2: C10 +1, C11 -1
        [(2, 1), (3, 1)],      # M3: C01 +1, C11 +1
        [(0, 1)],              # M4: C00 +1
        [(0, -1), (2, 1)],     # M5: C00 -1, C01 +1
        [(1, 1), (3, 1)],      # M6: C10 +1, C11 +1
        [(0, 1)],              # M7: C00 +1
    ]

    # Now build the full decomposition
    # A has 4x4 = 16 elements, indexed as i*m + j = i*4 + j
    # B has 4x6 = 24 elements, indexed as j*p + k = j*6 + k
    # C has 6x4 = 24 elements, indexed as k*n + i = k*4 + i
    #
    # For each (row_block, col_block, j_block, strassen_term):
    # row_block in {0, 1}, col_block in {0, 1, 2}, j_block in {0, 1}, strassen_term in {0..6}
    # Total: 2 * 3 * 2 * 7 = 84 terms

    rank = 2 * 3 * 2 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for row_block in range(2):
        for col_block in range(3):
            for j_block in range(2):
                for s in range(7):
                    # A block at (row_block, j_block): rows 2*row_block..2*row_block+1, cols 2*j_block..2*j_block+1
                    # A element (i, j) -> index i*4 + j
                    # For local (li, lj) in the 2x2 block: i = 2*row_block + li, j = 2*j_block + lj
                    for li in range(2):
                        for lj in range(2):
                            i = 2 * row_block + li
                            j = 2 * j_block + lj
                            U[i * m + j][col] = U_strassen[s][li * 2 + lj]

                    # B block at (j_block, col_block): rows 2*j_block..2*j_block+1, cols 2*col_block..2*col_block+1
                    # B element (j, k) -> index j*6 + k
                    # For local (lj, lk) in the 2x2 block: j = 2*j_block + lj, k = 2*col_block + lk
                    for lj in range(2):
                        for lk in range(2):
                            j = 2 * j_block + lj
                            k = 2 * col_block + lk
                            V[j * p + k][col] = V_strassen[s][lj * 2 + lk]

                    # C block at (row_block, col_block): rows 2*row_block..2*row_block+1, cols 2*col_block..2*col_block+1
                    # C element (i, k) -> index k*4 + i
                    # For local (li, lk) in the 2x2 block: i = 2*row_block + li, k = 2*col_block + lk
                    for (local_idx, coeff) in W_strassen_local[s]:
                        li, lk = local_idx // 2, local_idx % 2  # Wait, this is wrong
                        # Actually, the local indices are: 0=C00, 1=C10, 2=C01, 3=C11
                        # C00: i=0, k=0
                        # C10: i=1, k=0
                        # C01: i=0, k=1
                        # C11: i=1, k=1
                        li_map = [0, 1, 0, 1]
                        lk_map = [0, 0, 1, 1]
                        li = li_map[local_idx]
                        lk = lk_map[local_idx]
                        i = 2 * row_block + li
                        k = 2 * col_block + lk
                        W[k * n + i][col] += coeff

                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
