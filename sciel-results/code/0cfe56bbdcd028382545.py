"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition of rank 56 for 2x4 * 4x8.
    # We decompose the 4x4 middle block into four 2x2 blocks and apply
    # Strassen (rank 7 per 2x2x2 block) to each, giving 8*7 = 56 rank.
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56

    # Strassen formulas for 2x2 matrix multiplication:
    # C = A * B where A, B, C are 2x2.
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For our problem: A is 2x4, B is 4x8, C is 2x8.
    # We split the middle dimension (4) into two halves of 2, and the output
    # dimension (8) into four halves of 2.
    #
    # For block (br, bc) where br in {0,1} selects which pair of columns of A,
    # and bc in {0,1,2,3} selects which pair of columns of B:
    #   A_local is 2x2: A[i, br*2 + c] for i,c in {0,1}
    #   B_local is 2x2: B[br*2 + r, bc*2 + s] for r,s in {0,1}
    #   C_local is 2x2: C[i, bc*2 + s] for i,s in {0,1}
    #
    # The full C = sum over br of A[:, br*2:(br+1)*2] @ B[br*2:(br+1)*2, :]
    # But since A is only 2x4 and B is 4x8, the product is:
    #   C[i, k] = sum_{j=0}^{3} A[i, j] * B[j, k]
    #
    # We can think of this as two blocks in j: j in {0,1} and j in {2,3}.
    # For each block, we multiply a 2x2 (A slice) by a 2x8 (B slice), but we
    # further split the B slice into four 2x2 blocks.
    #
    # Actually, let's think more carefully. The tensor for 2x4 * 4x8 can be
    # seen as 4 independent 2x2 * 2x2 multiplications (one for each pair of
    # j values... no, that's not right either).
    #
    # Let me reconsider. The standard way: C[i,k] = sum_j A[i,j] B[j,k].
    # j ranges over 0,1,2,3. For each j, this is a rank-1 term.
    #
    # With Strassen, we group terms. The 2x4 * 4x8 problem is equivalent to
    # computing 2x8 = 8 outputs, each being a dot product of a length-2 vector
    # (row of A) with a length-4 vector... no, each output C[i,k] = sum_j A[i,j]*B[j,k].
    #
    # Better approach: View as block multiplication where we partition the
    # 4 middle indices into two groups of 2. Then:
    #   C = A_left * B_top + A_right * B_bottom
    # where A_left is 2x2 (columns 0,1), A_right is 2x2 (columns 2,3),
    # B_top is 2x8 (rows 0,1), B_bottom is 2x8 (rows 2,3).
    #
    # But each A_slice * B_slice is a 2x2 * 2x8 = 2x8 product.
    # We can further split B_slice into four 2x2 blocks (by columns).
    #
    # So for each of the two j-blocks (br=0,1) and each of the four k-blocks (bc=0,1,2,3):
    #   We compute a 2x2 * 2x2 product using Strassen (7 multiplications).
    #   Total: 2 * 4 * 7 = 56 rank.

    # For br in {0,1} and bc in {0,1,2,3}:
    #   A_local[i, c] = A[i, br*2 + c] for i,c in {0,1}
    #   B_local[r, s] = B[br*2 + r, bc*2 + s] for r,s in {0,1}
    #   C_local[i, s] contributes to C[i, bc*2 + s]
    #
    # Strassen indices (local):
    #   A00 = A[0, br*2+0], A01 = A[0, br*2+1], A10 = A[1, br*2+0], A11 = A[1, br*2+1]
    #   B00 = B[br*2+0, bc*2+0], B01 = B[br*2+0, bc*2+1]
    #   B10 = B[br*2+1, bc*2+0], B11 = B[br*2+1, bc*2+1]

    def idx_U(i, j): return i * m + j  # A index
    def idx_V(j, k): return j * p + k  # B index
    def idx_W(k, i): return k * n + i  # C index (trace convention W[k*n+i, r])

    col = 0
    for br in range(2):
        for bc in range(4):
            # 7 Strassen terms
            # M1: (A00+A11)*(B00+B11)
            #   A: A[0,br*2+0] + A[1,br*2+1]
            #   B: B[br*2+0,bc*2+0] + B[br*2+1,bc*2+1]
            #   C: contributes to C[0,bc*2+0] and C[1,bc*2+1]
            U[idx_U(0, br*2+0)][col] += 1
            U[idx_U(1, br*2+1)][col] += 1
            V[idx_V(br*2+0, bc*2+0)][col] += 1
            V[idx_V(br*2+1, bc*2+1)][col] += 1
            W[idx_W(bc*2+0, 0)][col] += 1
            W[idx_W(bc*2+1, 1)][col] += 1
            col += 1

            # M2: (A10+A11)*B00
            #   A: A[1,br*2+0] + A[1,br*2+1]
            #   B: B[br*2+0,bc*2+0]
            #   C: contributes to C[1,bc*2+0]
            U[idx_U(1, br*2+0)][col] += 1
            U[idx_U(1, br*2+1)][col] += 1
            V[idx_V(br*2+0, bc*2+0)][col] += 1
            W[idx_W(bc*2+0, 1)][col] += 1
            col += 1

            # M3: A00*(B01-B11)
            #   A: A[0,br*2+0]
            #   B: B[br*2+0,bc*2+1] - B[br*2+1,bc*2+1]
            #   C: contributes to C[0,bc*2+1]
            U[idx_U(0, br*2+0)][col] += 1
            V[idx_V(br*2+0, bc*2+1)][col] += 1
            V[idx_V(br*2+1, bc*2+1)][col] -= 1
            W[idx_W(bc*2+1, 0)][col] += 1
            col += 1

            # M4: A11*(B10-B00)
            #   A: A[1,br*2+1]
            #   B: B[br*2+1,bc*2+0] - B[br*2+0,bc*2+0]
            #   C: contributes to C[0,bc*2+0] and C[1,bc*2+0]
            U[idx_U(1, br*2+1)][col] += 1
            V[idx_V(br*2+1, bc*2+0)][col] += 1
            V[idx_V(br*2+0, bc*2+0)][col] -= 1
            W[idx_W(bc*2+0, 0)][col] += 1
            W[idx_W(bc*2+0, 1)][col] += 1
            col += 1

            # M5: (A00+A01)*B11
            #   A: A[0,br*2+0] + A[0,br*2+1]
            #   B: B[br*2+1,bc*2+1]
            #   C: contributes to C[0,bc*2+0] and C[0,bc*2+1]
            U[idx_U(0, br*2+0)][col] += 1
            U[idx_U(0, br*2+1)][col] += 1
            V[idx_V(br*2+1, bc*2+1)][col] += 1
            W[idx_W(bc*2+0, 0)][col] += 1
            W[idx_W(bc*2+1, 0)][col] += 1
            col += 1

            # M6: (A10-A00)*(B00+B01)
            #   A: A[1,br*2+0] - A[0,br*2+0]
            #   B: B[br*2+0,bc*2+0] + B[br*2+0,bc*2+1]
            #   C: contributes to C[1,bc*2+1]
            U[idx_U(1, br*2+0)][col] += 1
            U[idx_U(0, br*2+0)][col] -= 1
            V[idx_V(br*2+0, bc*2+0)][col] += 1
            V[idx_V(br*2+0, bc*2+1)][col] += 1
            W[idx_W(bc*2+1, 1)][col] += 1
            col += 1

            # M7: (A01-A11)*(B10+B11)
            #   A: A[0,br*2+1] - A[1,br*2+1]
            #   B: B[br*2+1,bc*2+0] + B[br*2+1,bc*2+1]
            #   C: contributes to C[0,bc*2+0] and C[1,bc*2+1]
            U[idx_U(0, br*2+1)][col] += 1
            U[idx_U(1, br*2+1)][col] -= 1
            V[idx_V(br*2+1, bc*2+0)][col] += 1
            V[idx_V(br*2+1, bc*2+1)][col] += 1
            W[idx_W(bc*2+0, 0)][col] += 1
            W[idx_W(bc*2+1, 1)][col] += 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
