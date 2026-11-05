"""Matrix multiplication 6x6x6 with block-Strassen decomposition (rank 189)."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # safety margin before 150s limit

    # Build the 6x6x6 tensor decomposition by dividing into 2x2 blocks.
    # 6x6 matrix = 3x3 blocks of 2x2.
    # For each block position (I,J) in the output 3x3 grid, the product of
    # 2x2 submatrices A[I,:] (which is A[I, J'] for the specific J) and
    # B[:,J] uses Strassen's 7-term algorithm.
    #
    # Actually, more carefully: for a 6x6 * 6x6 product with 2x2 blocks,
    # the output C is a 3x3 grid of 2x2 blocks.
    # Block (I,J) of C = sum over K of A_block(I,K) * B_block(K,J), each 2x2.
    # Strassen replaces the 8 scalar mults per 2x2 product with 7.
    # So instead of 3*3*8 = 72 rank-1 terms per block position, we use 3*3*7 = 63.
    # Total rank = 27 * 7 = 189.
    #
    # But wait: Strassen works on the 4 sub-elements of a 2x2 block product.
    # A 2x2 * 2x2 product: C = A*B where A,B are 2x2.
    # Strassen: 7 scalar multiplications, linear combinations for additions.
    # Each of the 7 scalar mults is a rank-1 term in the overall decomposition.
    #
    # Let me set up: 6x6 matrix indexed by (r,c) where r,c in 0..5.
    # Block structure: block_row = r//2, block_col = c//2, each 2x2.
    #
    # U has 36 rows (6*6), V has 36 rows (6*6), W has 36 rows (6*6).
    # For each output block (i_blk, j_blk) in 0..2, and each inner block k_blk in 0..2:
    #   The 2x2 product A[i_blk,k_blk] * B[k_blk,j_blk] contributes 7 rank-1 terms.
    #   Each term involves a linear combination of 4 elements of A[i_blk,k_blk]
    #   multiplied by a linear combination of 4 elements of B[k_blk,j_blk],
    #   and the result is added to 4 elements of C[i_blk,j_blk].
    #
    # Strassen's 7 products for X*Y (2x2 each):
    #   M1 = (x00+x11)*(y00+y11)
    #   M2 = (x10+x11)*y00
    #   M3 = x00*(y10-y11)
    #   M4 = x11*(y01-y00)
    #   M5 = (x00+x01)*y11
    #   M6 = (x10-x00)*(y00+y10)
    #   M7 = (x01-x11)*(y01+y11)
    #
    # Reconstruct:
    #   c00 = M1 + M4 - M5 + M7
    #   c01 = M3 + M5
    #   c10 = M2 + M4
    #   c11 = M1 - M2 + M3 + M6

    rank = 27 * 7  # 189
    U = np.zeros((36, rank), dtype=float)
    V = np.zeros((36, rank), dtype=float)
    W = np.zeros((36, rank), dtype=float)

    # Strassen factors: for each of the 7 terms, we have
    # alpha: 4 coefficients for X's 4 elements (row-major: 00,01,10,11)
    # beta:  4 coefficients for Y's 4 elements
    # gamma: 4 coefficients for C's 4 elements (which term contributes)

    # alpha_s[k][a] = coefficient of X[a] in product M_k
    # beta_s[k][b]  = coefficient of Y[b] in product M_k
    # gamma_s[t][k] = coefficient of M_k in C[t]

    alpha_s = [
        [1, 0, 0, 1],   # M1: x00 + x11
        [0, 0, 1, 1],   # M2: x10 + x11
        [1, 0, 0, 0],   # M3: x00
        [0, 0, 0, 1],   # M4: x11
        [1, 1, 0, 0],   # M5: x00 + x01
        [-1, 0, 1, 0],  # M6: x10 - x00
        [0, 1, 0, -1],  # M7: x01 - x11
    ]

    beta_s = [
        [1, 0, 0, 1],   # M1: y00 + y11
        [1, 0, 0, 0],   # M2: y00
        [0, 0, 1, -1],  # M3: y10 - y11
        [-1, 1, 0, 0],  # M4: y01 - y00
        [0, 0, 0, 1],   # M5: y11
        [1, 0, 1, 0],   # M6: y00 + y10
        [0, 1, 0, 1],   # M7: y01 + y11
    ]

    gamma_s = [
        [1, 0, 0, 1],   # c00: M1 + M4 - M5 + M7
        [0, 0, 1, 1],   # c01: M3 + M5
        [0, 1, 0, 1],   # c10: M2 + M4
        [1, -1, 1, 0],  # c11: M1 - M2 + M3 + M6
    ]

    # Wait, let me recheck gamma_s indexing.
    # gamma_s[c_element][m_term] = coefficient
    # c00 = 1*M1 + 0*M2 + 0*M3 + 1*M4 + (-1)*M5 + 0*M6 + 1*M7
    # c01 = 0*M1 + 0*M2 + 1*M3 + 0*M4 + 1*M5 + 0*M6 + 0*M7
    # c10 = 0*M1 + 1*M2 + 0*M3 + 1*M4 + 0*M5 + 0*M6 + 0*M7
    # c11 = 1*M1 + (-1)*M2 + 1*M3 + 0*M4 + 0*M5 + 1*M6 + 0*M7

    gamma_s = [
        [1, 0, 0, 1, -1, 0, 1],   # c00
        [0, 0, 1, 0, 1, 0, 0],    # c01
        [0, 1, 0, 1, 0, 0, 0],    # c10
        [1, -1, 1, 0, 0, 1, 0],   # c11
    ]

    # Now build the matrices.
    # Row index in U: i*n + j = i*6 + j (input A element at row i, col j)
    # Row index in V: j*m + k = j*6 + k (input B element at row j, col k)
    # Row index in W: k*n + i = k*6 + i (output C element at row k, col i) -- trace convention W[k*n+i, r]

    col = 0
    for i_blk in range(3):
        for k_blk in range(3):
            for j_blk in range(3):
                for mt in range(7):
                    # alpha_s[mt][a]: coefficient for X[a] where a indexes the 4 elements of A[i_blk, k_blk]
                    # X = A[i_blk, k_blk] is a 2x2 block of A.
                    # A element at block row i_blk, block col k_blk has local indices (a_row, a_col) in 0..1.
                    # Global row in A: i_blk*2 + a_row, global col: k_blk*2 + a_col.
                    # U row index: (i_blk*2 + a_row) * 6 + (k_blk*2 + a_col)

                    for a in range(4):
                        a_row = a // 2
                        a_col = a % 2
                        u_row = (i_blk * 2 + a_row) * 6 + (k_blk * 2 + a_col)
                        U[u_row, col] += alpha_s[mt][a]

                    # beta_s[mt][b]: coefficient for Y[b] where b indexes the 4 elements of B[k_blk, j_blk]
                    # Y = B[k_blk, j_blk] is a 2x2 block of B.
                    # B element at block row k_blk, block col j_blk has local indices (b_row, b_col) in 0..1.
                    # Global row in B: k_blk*2 + b_row, global col: j_blk*2 + b_col.
                    # V row index: (k_blk*2 + b_row) * 6 + (j_blk*2 + b_col)

                    for b in range(4):
                        b_row = b // 2
                        b_col = b % 2
                        v_row = (k_blk * 2 + b_row) * 6 + (j_blk * 2 + b_col)
                        V[v_row, col] += beta_s[mt][b]

                    # gamma_s[c][mt]: coefficient of M_mt in C[c]
                    # C element at block row i_blk, block col j_blk has local index c in 0..3.
                    # c_row = c // 2, c_col = c % 2
                    # Global row in C: i_blk*2 + c_row, global col: j_blk*2 + c_col.
                    # W row index (trace convention W[k*n+i]): row k = i_blk*2 + c_row, col i = j_blk*2 + c_col
                    # W row = (i_blk*2 + c_row) * 6 + (j_blk*2 + c_col)

                    for c in range(4):
                        c_row = c // 2
                        c_col = c % 2
                        w_row = (i_blk * 2 + c_row) * 6 + (j_blk * 2 + c_col)
                        W[w_row, col] += gamma_s[c][mt]

                    col += 1

    # Convert to lists of floats (integer values are fine)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    # Clean up: convert near-zero floats to exact ints
    def clean_list(lst):
        result = []
        for row in lst:
            new_row = []
            for val in row:
                if abs(val) < 1e-10:
                    new_row.append(0)
                elif abs(val - round(val)) < 1e-10:
                    new_row.append(int(round(val)))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result

    return {"U": clean_list(U_list), "V": clean_list(V_list), "W": clean_list(W_list)}
# EVOLVE-BLOCK-END
