import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def _strassen_4x4():
    """Return (U, V, W) for 4x4 x 4x4 -> 4x4 rank-17 Strassen.
    U: (16,17), V: (16,17), W: (16,17) with W[k*4+i, r].
    """
    rank = 17
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Standard Strassen for 4x4: split into 2x2 blocks
    # A = [[A00,A01],[A10,A11]] each 2x2
    # B = [[B00,B01],[B10,B11]] each 2x2
    # C = [[C00,C01],[C10,C11]] each 2x2
    # 
    # M1 = A00*(B00+B10)
    # M2 = (A00+A01)*B10
    # M3 = (A10+A11)*B00
    # M4 = A11*(B10+B11)
    # M5 = (A00+A11)*(B00+B11)
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A00-A10)*(B00+B10)
    #
    # C00 = M5 + M4 - M1 - M7
    # C01 = M1 + M2
    # C10 = M3 + M4
    # C11 = M5 + M6 - M2 - M3
    #
    # But each M is a 2x2 product, which itself needs rank. For 4x4 we can
    # use the 7 Strassen products where each "scalar" multiplication is replaced
    # by a 2x2 schoolbook (4 multiplications), giving 7*4=28. That's not rank 17.
    #
    # Actually, the rank of 4x4 matrix multiplication is not 17. Strassen gives
    # log_2(7) per level, so for 4x4 it's 7^2 = 49? No.
    #
    # Let me reconsider. The rank of n x n matrix multiplication:
    # n=2: rank 7 (Strassen)
    # n=4: rank at most 7^2 = 49 (applying Strassen twice)
    # But the actual rank of 4x4x4 tensor is not 17.
    #
    # The strategy says "known rank-17 exact decomposition for 4x4x4". Let me
    # check: the border rank of 4x4x4 is 17 (Landsberg). But the actual rank
    # is at least 24 and at most 49. So rank 17 is NOT achievable exactly.
    #
    # This strategy is flawed. Rank 17 is the border rank, not the rank.
    # We cannot get an exact decomposition of rank 17 for 4x4x4.
    #
    # Let me instead just return the current working 42-rank solution.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    steps = [
        (
            {(0,0): 1, (0,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1), (3, 1)]
        ),
        (
            {(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1},
            [(2, 1), (3, -1)]
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(0,3): 1, (0,4): 1, (0,5): 1, (1,3): 1, (1,4): 1, (1,5): 1, (2,3): -1, (2,4): -1, (2,5): -1, (3,3): -1, (3,4): -1, (3,5): -1},
            [(1, 1), (3, 1)]
        ),
        (
            {(1,2): 1, (1,3): 1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (3,0): 1, (3,1): 1, (3,2): 1, (0,0): -1, (0,1): -1, (0,2): -1, (1,0): -1, (1,1): -1, (1,2): -1},
            [(0, 1), (2, 1)]
        ),
        (
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1},
            {(2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, -1), (1, 1)]
        ),
        (
            {(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (0,4): 1, (0,5): 1, (1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1, (1,4): 1, (1,5): 1},
            [(3, 1)]
        ),
        (
            {(0,2): 1, (0,3): 1, (1,2): -1, (1,3): -1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,0): 1, (3,1): 1, (3,2): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1)]
        ),
    ]

    col = 0
    for X_dict, Y_dict, C_targets in steps:
        x0_coeffs = {}
        x1_coeffs = {}
        for (i, j), c in X_dict.items():
            if j % 2 == 0:
                x0_coeffs[(i, j)] = c
            else:
                x1_coeffs[(i, j)] = c

        y_coeffs = [[], [], [], [], [], []]
        for (j, k), c in Y_dict.items():
            r_block = j % 2
            c_block = k % 3
            idx = r_block * 3 + c_block
            y_coeffs[idx].append(((j, k), c))

        sub_terms = [
            (x0_coeffs, y_coeffs[0], 0),
            (x1_coeffs, y_coeffs[3], 0),
            (x0_coeffs, y_coeffs[1], 1),
            (x1_coeffs, y_coeffs[4], 1),
            (x0_coeffs, y_coeffs[2], 2),
            (x1_coeffs, y_coeffs[5], 2),
        ]

        for X_c, Y_c, z_idx in sub_terms:
            for (i, j), c in X_c.items():
                U[i * m + j, col] += c
            for (j, k), c in Y_c:
                V[j * p + k, col] += c

            for block_idx, sign in C_targets:
                if block_idx == 0:
                    i_C = 0
                    k_C = z_idx
                elif block_idx == 1:
                    i_C = 0
                    k_C = 3 + z_idx
                elif block_idx == 2:
                    i_C = 1
                    k_C = z_idx
                elif block_idx == 3:
                    i_C = 1
                    k_C = 3 + z_idx

                W[k_C * n + i_C, col] += sign

            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
