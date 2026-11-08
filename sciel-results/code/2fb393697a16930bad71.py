import numpy as np
import time

def construct():
    """
    Matrix multiplication 3x4 by 4x4.
    Uses rank 44 construction from splitting 4x4 into four 2x2 blocks
    and applying an 11-rank decomposition for 3x2 * 2x2 -> 3x2.
    Total rank = 4 * 11 = 44.
    """
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill

    n, m, p = 3, 4, 4
    rank = 44

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Local factors for 3x2x2 multiplication
    # A is 3x2, B is 2x2, C is 3x2
    # Local A indices: 0..5 (row * 2 + col)
    # Local B indices: 0..3 (row * 2 + col)
    # Local C indices: 0..5 (row * 2 + col)

    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))

    # Strassen part (cols 0-6) for rows 0,1 of A
    # M1: (a00+a10)(b00+b10)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1

    # M2: a10 * b00
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1

    # M3: a00 * (b10-b00)
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1

    # M4: a10 * b10
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1

    # M5: (a00+a01) * b01
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1

    # M6: (a10-a00) * (b00+b11)
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1

    # M7: (a01-a10) * (b10+b11)
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1

    # Extra part (cols 7-10) for Row 2 of A
    # M8: a20 * b00 -> c20
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1

    # M9: a21 * b10 -> c20
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1

    # M10: a20 * b01 -> c21
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1

    # M11: a21 * b11 -> c21
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1

    # Mapping function to place local block into global matrices
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11

        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                U[global_a_idx, col_offset:col_offset + 11] += U_loc[local_a_idx, :]

        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                V[global_b_idx, col_offset:col_offset + 11] += V_loc[local_b_idx, :]

        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                # W uses trace convention: W[k*n+i, r] where k=col, i=row
                global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                W[global_c_idx, col_offset:col_offset + 11] += W_loc[local_c_idx, :]

    # Block 0: A[0:3, 0:2] * B[0:2, 0:2] -> C[0:3, 0:2]
    add_block(0, 0, 0, 0, 0, 0, 0)

    # Block 1: A[0:3, 2:4] * B[2:4, 0:2] -> C[0:3, 0:2]
    add_block(1, 0, 2, 2, 0, 0, 0)

    # Block 2: A[0:3, 0:2] * B[0:2, 2:4] -> C[0:3, 2:4]
    add_block(2, 0, 0, 0, 2, 0, 2)

    # Block 3: A[0:3, 2:4] * B[2:4, 2:4] -> C[0:3, 2:4]
    add_block(3, 0, 2, 2, 2, 0, 2)

    # Deadline guard
    if time.time() - start_time > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
