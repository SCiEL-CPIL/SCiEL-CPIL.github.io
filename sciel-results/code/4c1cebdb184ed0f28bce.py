import time
import numpy as np

DIMENSIONS = [2, 4, 8]


def construct():
    start_time = time.time()
    DEADLINE = start_time + 130  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # A is n x m (2x4), B is m x p (4x8), C is n x p (2x8)

    # --- Strassen 2x2 * 2x2 rank-7 base tensor (local indices 0..3) ---
    # local layout: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)

    # M0: (A00+A11)*(B00+B11) -> C00, C11
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1; W_local[3, 0] = 1
    # M1: (A10+A11)*B00 -> C10, C11
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1; W_local[3, 1] = -1
    # M2: A00*(B01-B11) -> C01, C11
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1; W_local[3, 2] = 1
    # M3: A11*(B10-B00) -> C00, C10
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1; W_local[2, 3] = 1
    # M4: (A00+A01)*B11 -> C00, C01
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1; W_local[1, 4] = 1
    # M5: (A10-A00)*(B00+B01) -> C11
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1
    # M6: (A01-A11)*(B10+B11) -> C00
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1

    # --- Kronecker with identity (rank 2) to build 2x2 * 2x4 block tensor ---
    # A block is 2x4 (a: 0..3), B block is 2x4 (b: 0..3), C block is 2x4 (c: 0..3)
    # Kronecker rank = 7 * 2 = 14
    total_rank = 14
    U_block = np.zeros((4, total_rank), dtype=int)
    V_block = np.zeros((4, total_rank), dtype=int)
    W_block = np.zeros((4, total_rank), dtype=int)

    for e in range(2):  # identity index e
        for l in range(7):
            r = e * 7 + l
            # A contribution: e selects the column-pair of the 2x4 A
            for la in range(4):
                val = U_local[la, l]
                if val != 0:
                    U_block[la, r] += val
            # B contribution: e selects the column-pair of the 2x4 B
            for lb in range(4):
                val = V_local[lb, l]
                if val != 0:
                    V_block[lb, r] += val
            # C contribution: e selects the column-pair of the 2x4 C
            for lc in range(4):
                val = W_local[lc, l]
                if val != 0:
                    W_block[lc, r] += val

    # --- Assemble full 2x4 * 4x8 tensor as two independent blocks ---
    # Block 0: A rows 0-1, A cols 0-1 ; B rows 0-1, B cols 0-3 ; C rows 0-1, C cols 0-3
    # Block 1: A rows 0-1, A cols 2-3 ; B rows 2-3, B cols 4-7 ; C rows 0-1, C cols 4-7
    full_rank = 2 * total_rank  # 28
    U = np.zeros((n * m, full_rank), dtype=int)   # 8 x 28
    V = np.zeros((m * p, full_rank), dtype=int)   # 32 x 28
    W = np.zeros((p * n, full_rank), dtype=int)   # 16 x 28

    for blk in range(2):
        base_r = blk * total_rank
        # A indices (flattened row-major, n x m = 2 x 4): row*4 + col
        a_map = [
            0 * 4 + blk * 2 + 0,   # (0, blk*2)
            0 * 4 + blk * 2 + 1,   # (0, blk*2+1)
            1 * 4 + blk * 2 + 0,   # (1, blk*2)
            1 * 4 + blk * 2 + 1,   # (1, blk*2+1)
        ]
        # B indices (m x p = 4 x 8): row*8 + col
        b_map = [
            (blk * 2 + 0) * 8 + blk * 4 + 0,   # (blk*2, blk*4)
            (blk * 2 + 0) * 8 + blk * 4 + 1,   # (blk*2, blk*4+1)
            (blk * 2 + 1) * 8 + blk * 4 + 0,   # (blk*2+1, blk*4)
            (blk * 2 + 1) * 8 + blk * 4 + 1,   # (blk*2+1, blk*4+1)
        ]
        # C indices (official convention W[k*n + i, r]: col*2 + row)
        c_map = [
            (blk * 4 + 0) * 2 + 0,   # (col blk*4, row 0)
            (blk * 4 + 1) * 2 + 0,   # (col blk*4+1, row 0)
            (blk * 4 + 0) * 2 + 1,   # (col blk*4, row 1)
            (blk * 4 + 1) * 2 + 1,   # (col blk*4+1, row 1)
        ]

        for r in range(total_rank):
            fr = base_r + r
            for la in range(4):
                val = U_block[la, r]
                if val != 0:
                    U[a_map[la], fr] += val
            for lb in range(4):
                val = V_block[lb, r]
                if val != 0:
                    V[b_map[lb], fr] += val
            for lc in range(4):
                val = W_block[lc, r]
                if val != 0:
                    W[c_map[lc], fr] += val

    if time.time() > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
