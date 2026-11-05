"""Matrix multiplication 2,3,8: Strassen-based rank-44 decomposition."""
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8

    # We split m=3 into two parts: m1=2, m2=1
    # For each k (0..7), the 2x1x2 block (rows i=0,1; cols j=0,1) uses Strassen (rank 7)
    # The 2x1x1 block (cols j=2) uses 2 multiplications (rank 2)
    # Total per k: 7 + 2 = 9
    # Total rank: 8 * 9 = 72... that's worse.

    # Let me reconsider. The m=3 dimension: we can group as 2+1.
    # For the 2x2x2 part (i=0,1; j=0,1; k=0,1): Strassen gives rank 7 for a full 2x2x2 block.
    # But k goes 0..7, so we have 8 different k values.

    # Alternative approach: split the problem into:
    # Part A: A[:, 0:2] @ B[0:2, :] -> C_part_A (2x8)
    # Part B: A[:, 2:3] @ B[2:3, :] -> C_part_B (2x8)
    # C = C_part_A + C_part_B

    # For Part B: A_col (2x1) @ B_rows (1x8) -> outer product, rank = 2*1*8 = 16.
    # That's 16 multiplications.

    # For Part A: A_left (2x2) @ B_left (2x8)
    # Split B_left into 4 blocks of 2x2: B_left[:, 0:2], B_left[:, 2:4], B_left[:, 4:6], B_left[:, 6:8]
    # Each 2x2x2 block: Strassen rank 7
    # Total: 4 * 7 = 28

    # Grand total: 28 + 16 = 44. This matches the known result!

    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]   # 6 x 44
    V = [[0] * rank_total for _ in range(m * p)]   # 24 x 44
    W = [[0] * rank_total for _ in range(p * n)]   # 16 x 44

    col = 0

    # Part A: 2x2x2 Strassen blocks for each of 4 column-pairs of B
    # A_left = A[0:2, 0:2], indexed as A00, A01, A10, A11 (i=0,1; j=0,1)
    # For block b (0..3), B_block = B[0:2, 2b:2b+2], indexed as B00, B01, B10, B11 (j=0,1; k=2b, 2b+1)
    # Strassen:
    # M1 = (A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A00 * (B00 + B10)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B01 + B11)
    # M7 = (A01 - A11) * (B00 + B10)
    # Results:
    # C00 = M5 + M3 - M7 - M1
    # C01 = M4 + M7
    # C10 = M2 + M6
    # C11 = M1 + M5 - M2 - M4

    strassen_defs = [
        # (U_indices, V_indices, W_indices, sign)
        # Each term: sum of (u_row, u_coeff) * (v_row, v_coeff) contributes to (w_row, w_coeff)
        # M1 = A11 * (B00 + B11)
        #   U: row A11 = i=1,j=1 -> index 1*2+1 = 3
        #   V: rows B00 (j=0,k=0 -> 0*2+0=0) and B11 (j=1,k=1 -> 1*2+1=3)
        #   W: C00 (k=0,i=0 -> 0*2+0=0) coeff -1; C11 (k=1,i=1 -> 1*2+1=3) coeff +1
        ([(3, 1)], [(0, 1), (3, 1)], [(0, -1), (3, 1)]),
        # M2 = (A10 + A11) * B00
        #   U: rows A10 (i=1,j=0 -> 1*2+0=2) and A11 (i=1,j=1 -> 3)
        #   V: row B00 (j=0,k=0 -> 0)
        #   W: C10 (k=0,i=1 -> 0*2+1=1) coeff +1
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1)]),
        # M3 = A00 * (B01 - B11)
        #   U: row A00 (i=0,j=0 -> 0)
        #   V: rows B01 (j=0,k=1 -> 0*2+1=1) coeff +1, B11 (j=1,k=1 -> 3) coeff -1
        #   W: C00 (k=0,i=0 -> 0) coeff +1; C11 (k=1,i=1 -> 3) coeff -1
        ([(0, 1)], [(1, 1), (3, -1)], [(0, 1), (3, -1)]),
        # M4 = A00 * (B00 + B10)
        #   U: row A00 (0)
        #   V: rows B00 (0) coeff +1, B10 (j=1,k=0 -> 1*2+0=2) coeff +1
        #   W: C01 (k=1,i=0 -> 1*2+0=2) coeff +1; C11 (k=1,i=1 -> 3) coeff -1
        ([(0, 1)], [(0, 1), (2, 1)], [(2, 1), (3, -1)]),
        # M5 = (A00 + A01) * B11
        #   U: rows A00 (0) coeff +1, A01 (i=0,j=1 -> 1) coeff +1
        #   V: row B11 (3)
        #   W: C00 (0) coeff +1; C11 (3) coeff +1
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (3, 1)]),
        # M6 = (A10 - A00) * (B01 + B11)
        #   U: rows A10 (2) coeff +1, A00 (0) coeff -1
        #   V: rows B01 (1) coeff +1, B11 (3) coeff +1
        #   W: C10 (1) coeff +1
        ([(2, 1), (0, -1)], [(1, 1), (3, 1)], [(1, 1)]),
        # M7 = (A01 - A11) * (B00 + B10)
        #   U: rows A01 (1) coeff +1, A11 (3) coeff -1
        #   V: rows B00 (0) coeff +1, B10 (2) coeff +1
        #   W: C01 (2) coeff +1; C00 (0) coeff -1
        ([(1, 1), (3, -1)], [(0, 1), (2, 1)], [(2, 1), (0, -1)]),
    ]

    for b in range(4):
        # B block offset: j is 0,1; k is 2b, 2b+1
        # B_row(j, k) = j * 8 + k  (since m=3, p=8, so B index = j*p + k)
        k0 = 2 * b
        k1 = 2 * b + 1
        # Map local B indices to global
        # Local (j=0, k=0) -> global (j=0, k=k0) -> 0*8 + k0 = k0
        # Local (j=0, k=1) -> global (j=0, k=k1) -> 0*8 + k1 = k1
        # Local (j=1, k=0) -> global (j=1, k=k0) -> 1*8 + k0 = 8+k0
        # Local (j=1, k=1) -> global (j=1, k=k1) -> 1*8 + k1 = 8+k1

        b_offsets = {
            0: k0,     # B00
            1: k1,     # B01
            2: 8 + k0, # B10
            3: 8 + k1, # B11
        }
        # Local B index mapping: 0->B00, 1->B01, 2->B10, 3->B11

        for u_terms, v_terms, w_terms in strassen_defs:
            for (u_row, u_coeff) in u_terms:
                # u_row is already global (i*m + j for i in 0,1 and j in 0,1)
                U[u_row][col] += u_coeff
            for (v_local, v_coeff) in v_terms:
                v_global = b_offsets[v_local]
                V[v_global][col] += v_coeff
            for (w_local, w_coeff) in w_terms:
                # w_local: 0->C00(k=k0,i=0), 1->C10(k=k0,i=1), 2->C01(k=k1,i=0), 3->C11(k=k1,i=1)
                # W index = k*n + i = k*2 + i
                if w_local == 0:
                    w_global = k0 * 2 + 0
                elif w_local == 1:
                    w_global = k0 * 2 + 1
                elif w_local == 2:
                    w_global = k1 * 2 + 0
                elif w_local == 3:
                    w_global = k1 * 2 + 1
                W[w_global][col] += w_coeff
            col += 1

    # Part B: A[:, 2:3] @ B[2:3, :] -> outer product of a 2-vector and 8-vector
    # A_col: i=0 -> row 2, i=1 -> row 3
    # B_rows: j=2, k=0..7 -> indices 2*8+k = 16+k
    # C: k*n+i = k*2+i
    # For each (i, k): one multiplication
    for i in range(2):
        for k in range(8):
            U[2 + i][col] = 1          # A[i, 2]
            V[16 + k][col] = 1         # B[2, k]
            W[k * 2 + i][col] = 1      # C[k, i]
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
