"""Rewritten program: Kronecker/logic decomposition for 4x4 * 4x8 using rank-7 2x2 multiplication."""
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # We exploit p = 2*m. The multiplication A(4x4) * B(4x8) can be seen as
    # two independent 4x4 * 4x4 multiplications (B split into B1, B2),
    # where C1 = A*B1 and C2 = A*B2.
    #
    # We use the 2x2 Strassen decomposition (rank 7) to build a rank-49
    # decomposition for 4x4 matrix multiplication, then duplicate it
    # for the two halves of B to get rank 98 for 4x4 * 4x8.
    #
    # 2x2 Strassen:
    #   A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    #   C = [[C00, C01], [C10, C11]]
    #
    #   m0 = (a + d)(e + h)
    #   m1 = (c + d)e
    #   m2 = a(f - h)
    #   m3 = d(g - e)
    #   m4 = (a + b)h
    #   m5 = (c - a)(e + f)
    #   m6 = (b - d)(g + h)
    #
    #   C00 = m0 + m3 - m4 + m6
    #   C01 = m2 + m4
    #   C10 = m1 + m3
    #   C11 = m0 - m1 + m2 + m5
    #
    # For the 4x4 case, we treat A and B as 2x2 blocks of 2x2 matrices,
    # and apply Strassen at the block level. This gives 7 block multiplications,
    # each of which is a 2x2 * 2x2 multiplication, itself done via Strassen
    # (rank 7). So total rank = 7 * 7 = 49 for 4x4 * 4x4.
    #
    # For 4x4 * 4x8: B is 4x8 = two 4x4 blocks. We do two 4x4*4x4
    # multiplications, each of rank 49, total rank 98.
    #
    # The tensor decomposition:
    #   U has shape (n*m, rank) = (16, 98)
    #   V has shape (m*p, rank) = (32, 98)
    #   W has shape (p*n, rank) = (32, 98)
    #
    # Convention: W[k*n + i, r]

    n, m, p = DIMENSIONS
    # m=4, p=8, n=4
    # B is 4x8, split into B_left (4x4) and B_right (4x4)
    # B_left columns 0..3, B_right columns 4..7

    # Build the 2x2 Strassen factors
    # For 2x2: U2 (4x7), V2 (4x7), W2 (4x7)
    # A indexed as [a,b,c,d] -> index 0=a, 1=b, 2=c, 3=d
    # B indexed as [e,f,g,h] -> index 0=e, 1=f, 2=g, 3=h
    # C indexed as [C00,C01,C10,C11] -> index 0=C00, 1=C01, 2=C10, 3=C11

    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # m0 = (a+d)(e+h)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # C00 += m0, C11 += m0
    W2[0, 0] = 1; W2[3, 0] = 1

    # m1 = (c+d)*e
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    # C10 += m1, C11 -= m1
    W2[2, 1] = 1; W2[3, 1] = -1

    # m2 = a*(f-h)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    # C01 += m2, C11 += m2
    W2[1, 2] = 1; W2[3, 2] = 1

    # m3 = d*(g-e)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    # C00 += m3, C10 += m3
    W2[0, 3] = 1; W2[2, 3] = 1

    # m4 = (a+b)*h
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    # C00 -= m4, C01 += m4
    W2[0, 4] = -1; W2[1, 4] = 1

    # m5 = (c-a)*(e+f)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    # C11 += m5
    W2[3, 5] = 1

    # m6 = (b-d)*(g+h)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    # C00 += m6
    W2[0, 6] = 1

    # Now build 4x4 decomposition using block Strassen.
    # A is 4x4, viewed as 2x2 blocks of 2x2:
    #   A = [[A00, A01], [A10, A11]]
    # B is 4x4, viewed as 2x2 blocks of 2x2:
    #   B = [[B00, B01], [B10, B11]]
    # C = A*B, 4x4, viewed as 2x2 blocks of 2x2:
    #   C = [[C00, C01], [C10, C11]]
    #
    # Block Strassen:
    #   M0 = (A00+A11)*(B00+B11)
    #   M1 = (A10+A11)*B00
    #   M2 = A00*(B01-B11)
    #   M3 = A11*(B10-B00)
    #   M4 = (A00+A01)*B11
    #   M5 = (A10-A00)*(B00+B01)
    #   M6 = (A01-A11)*(B10+B11)
    #
    #   C00 = M0 + M3 - M4 + M6
    #   C01 = M2 + M4
    #   C10 = M1 + M3
    #   C11 = M0 - M1 + M2 + M5
    #
    # Each Mi is a 2x2 * 2x2 product, decomposed via Strassen (rank 7).
    # So each Mi contributes 7 rank-1 terms.
    # Total: 7*7 = 49 terms for 4x4*4x4.

    # For each of the 7 block-level Strassen terms (block term bt),
    # we have a product of two 2x2 matrices, which we decompose via 2x2 Strassen
    # (7 inner terms).
    #
    # U4 (16 x 49): For block term bt, inner term it:
    #   The A-side 2x2 submatrix is determined by bt.
    #   Within that 2x2, the inner Strassen term it determines which
    #   2x2 entries of the submatrix are used.
    #
    # Let's define the block-level U and V coefficients:
    # For each block term bt (0..6), the A-side 2x2 submatrix has a
    # 4-element vector u_bt (coefficients for A00, A01, A10, A11 entries
    # at the block level). But actually, each block is a full 2x2, so
    # the "submatrix" for block bt on the A side is a linear combination
    # of the 4 blocks, each contributing a full 2x2.
    #
    # More precisely:
    #   For block term bt, the A-side "matrix" (4x4) has blocks:
    #     block (0,0): coeff_a00[bt] * A00
    #     block (0,1): coeff_a01[bt] * A01
    #     block (1,0): coeff_a10[bt] * A10
    #     block (1,1): coeff_a11[bt] * A11
    #
    #   Similarly for B-side.
    #
    # The 4x4 A matrix has entries:
    #   A[row, col] where row = bi*2 + ri, col = bj*2 + cj
    #   bi, bj in {0,1} are block indices, ri, cj in {0,1} are within-block.
    #
    # For block term bt, the contribution to U4[row, col, (bt*7 + it)] is:
    #   coeff_a[bt][bi, bj] * U2[ri, cj, it]
    #
    # where coeff_a[bt] is a 2x2 matrix of coefficients for the A blocks.

    # Block-level A coefficients (2x2 matrix for each bt):
    # M0: A00+A11 -> [[1,0],[0,1]]
    # M1: A10+A11 -> [[0,0],[1,1]]
    # M2: A00      -> [[1,0],[0,0]]
    # M3: A11      -> [[0,0],[0,1]]
    # M4: A00+A01  -> [[1,1],[0,0]]
    # M5: A10-A00  -> [[-1,0],[1,0]]
    # M6: A01-A11  -> [[0,-1],[0,1]]

    a_block = [
        np.array([[1, 0], [0, 1]]),   # M0
        np.array([[0, 0], [1, 1]]),   # M1
        np.array([[1, 0], [0, 0]]),   # M2
        np.array([[0, 0], [0, 1]]),   # M3
        np.array([[1, 1], [0, 0]]),   # M4
        np.array([[-1, 0], [1, 0]]),  # M5
        np.array([[0, -1], [0, 1]]),  # M6
    ]

    # Block-level B coefficients (2x2 matrix for each bt):
    # M0: B00+B11 -> [[1,0],[0,1]]
    # M1: B00      -> [[1,0],[0,0]]
    # M2: B01-B11  -> [[0,1],[0,-1]]
    # M3: B10-B00  -> [[-1,0],[1,0]]
    # M4: B11      -> [[0,0],[0,1]]
    # M5: B00+B01  -> [[1,1],[0,0]]
    # M6: B10+B11  -> [[0,1],[1,1]]

    b_block = [
        np.array([[1, 0], [0, 1]]),   # M0
        np.array([[1, 0], [0, 0]]),   # M1
        np.array([[0, 1], [0, -1]]),  # M2
        np.array([[-1, 0], [1, 0]]),  # M3
        np.array([[0, 0], [0, 1]]),   # M4
        np.array([[1, 1], [0, 0]]),   # M5
        np.array([[0, 1], [1, 1]]),   # M6
    ]

    # C-side: for each block term bt, the contribution to C blocks:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    #
    # c_block[bt][ci, cj] is the coefficient of block (ci, cj) of C
    # that comes from block term bt.
    # ci, cj in {0,1} are block indices of C.

    c_block = [
        np.array([[1, 0], [0, 1]]),   # M0: C00 += M0, C11 += M0
        np.array([[0, 0], [1, -1]]),  # M1: C10 += M1, C11 -= M1
        np.array([[0, 1], [0, 1]]),   # M2: C01 += M2, C11 += M2
        np.array([[1, 0], [1, 0]]),   # M3: C00 += M3, C10 += M3
        np.array([[-1, 1], [0, 0]]),  # M4: C00 -= M4, C01 += M4
        np.array([[0, 0], [0, 1]]),   # M5: C11 += M5
        np.array([[1, 0], [0, 0]]),   # M6: C00 += M6
    ]

    # Build U4 (16 x 49), V4 (16 x 49), W4 (16 x 49) for 4x4*4x4
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))

    for bt in range(7):
        for it in range(7):
            col = bt * 7 + it
            # U4: row = bi*2 + ri, where bi is block row of A, ri is within-block row
            # For block term bt, A block (bi, bj) has coefficient a_block[bt][bi, bj]
            # Within that 2x2 block, U2[ri, cj, it] gives the coefficient for
            # entry (ri, cj) of the 2x2 submatrix.
            for bi in range(2):
                for bj in range(2):
                    coeff_a = a_block[bt][bi, bj]
                    if coeff_a != 0:
                        for ri in range(2):
                            for cj in range(2):
                                u_entry = coeff_a * U2[ri, cj, it]
                                if u_entry != 0:
                                    row = bi * 2 + ri
                                    col_idx = bj * 2 + cj
                                    U4[row * 4 + col_idx, col] += u_entry

            # V4: similar for B
            for bi in range(2):
                for bj in range(2):
                    coeff_b = b_block[bt][bi, bj]
                    if coeff_b != 0:
                        for ri in range(2):
                            for cj in range(2):
                                v_entry = coeff_b * V2[ri, cj, it]
                                if v_entry != 0:
                                    row = bi * 2 + ri
                                    col_idx = bj * 2 + cj
                                    V4[row * 4 + col_idx, col] += v_entry

            # W4: for C, block (ci, cj) has coefficient c_block[bt][ci, cj]
            # Within that 2x2 block, W2[ri, cj, it] gives the coefficient.
            for ci in range(2):
                for cj in range(2):
                    coeff_c = c_block[bt][ci, cj]
                    if coeff_c != 0:
                        for ri in range(2):
                            for cj2 in range(2):
                                w_entry = coeff_c * W2[ri, cj2, it]
                                if w_entry != 0:
                                    # C entry at block (ci, cj), within-block (ri, cj2)
                                    # C is 4x4, indexed as [row, col]
                                    row = ci * 2 + ri
                                    col_idx = cj * 2 + cj2
                                    # W convention: W[k*n + i, r] where k is col of C, i is row of C
                                    # So index = col_idx * n + row = col_idx * 4 + row
                                    W4[col_idx * 4 + row, col] += w_entry

    # Now for 4x4 * 4x8:
    # B is 4x8, split into B1 (cols 0-3) and B2 (cols 4-7).
    # C1 = A * B1 (4x4), C2 = A * B2 (4x4).
    # C = [C1 | C2] (4x8).
    #
    # For B1: V1[j, k] for k in 0..3, maps to V4[j * 4 + k, col]
    # For B2: V2[j, k] for k in 4..7, maps to V4[j * 4 + (k-4), col + 49]
    #
    # W for C1: W1[k, i] for k in 0..3, maps to W4[k * 4 + i, col]
    # W for C2: W2[k, i] for k in 4..7, maps to W4[(k-4) * 4 + i, col + 49]
    #
    # U is the same for both: U4.

    total_rank = 98
    U = np.zeros((16, total_rank))
    V = np.zeros((32, total_rank))
    W = np.zeros((32, total_rank))

    # U: same for both halves
    U[:, 0:49] = U4
    U[:, 49:98] = U4

    # V: first 49 columns use V4 for B1 (B cols 0-3)
    # V4 has shape (16, 49), where row = j*4 + k for j in 0..3 (rows of B), k in 0..3 (cols of B)
    # B1: B[j, k] for j in 0..3, k in 0..3
    # V index: j * p + k = j * 8 + k
    for j in range(4):
        for k in range(4):
            v_row = j * 8 + k
            V[v_row, 0:49] = V4[j * 4 + k, 0:49]

    # V: second 49 columns use V4 for B2 (B cols 4-7)
    # B2: B[j, k] for j in 0..3, k in 4..7
    # V index: j * 8 + k
    for j in range(4):
        for k in range(4):
            v_row = j * 8 + (k + 4)
            V[v_row, 49:98] = V4[j * 4 + k, 0:49]

    # W: first 49 columns for C1 (C cols 0-3)
    # W4 has shape (16, 49), where row = k*4 + i for k in 0..3 (cols of C), i in 0..3 (rows of C)
    # W convention: W[k*n + i, r] = W[k*4 + i, r]
    for k in range(4):
        for i in range(4):
            w_row = k * 4 + i
            W[w_row, 0:49] = W4[k * 4 + i, 0:49]

    # W: second 49 columns for C2 (C cols 4-7)
    for k in range(4):
        for i in range(4):
            w_row = (k + 4) * 4 + i
            W[w_row, 49:98] = W4[k * 4 + i, 0:49]

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
