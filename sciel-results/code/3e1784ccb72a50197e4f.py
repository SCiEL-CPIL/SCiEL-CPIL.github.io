import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    deadline = time.time() + 130  # Safety margin before 150s kill

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
    # M1: (a00+a10)(b00+b10) -> c00, c11
    U_loc[0, 0] = 1; U_loc[2, 0] = 1
    V_loc[0, 0] = 1; V_loc[2, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1

    # M2: (a10-a00)b00 -> c10, c11
    U_loc[2, 1] = 1; U_loc[0, 1] = -1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1

    # M3: a01(b10-b00) -> c01, c11
    U_loc[0, 2] = 1; U_loc[1, 2] = 1
    V_loc[2, 2] = 1; V_loc[0, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1

    # M4: a10(b01-b11) -> c00, c10
    U_loc[2, 3] = 1
    V_loc[1, 3] = 1; V_loc[3, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1

    # M5: (a00+a01)b11 -> c00, c01
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1

    # M6: (a11-a01)(b00+b11) -> c11
    U_loc[3, 5] = 1; U_loc[1, 5] = -1
    V_loc[0, 5] = 1; V_loc[3, 5] = 1
    W_loc[3, 5] = 1

    # M7: (a01+a10)(b10-b11) -> c00
    U_loc[1, 6] = 1; U_loc[2, 6] = 1
    V_loc[2, 6] = 1; V_loc[3, 6] = -1
    W_loc[0, 6] = 1

    # Trivial part (cols 7-10) for row 2 of A
    # M8: a20*b00 -> c20
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1

    # M9: a21*b10 -> c20
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1

    # M10: a20*b01 -> c21
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1

    # M11: a21*b11 -> c21
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1

    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11

        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]

        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]

        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

    # 4 blocks: (A_cols 0-1 or 2-3) x (B_rows 0-1 or 2-3) -> C_cols 0-1 or 2-3
    add_block(0, 0, 0, 0, 0, 0, 0)   # A[:,0:2] x B[0:2,:] -> C[:,0:2]
    add_block(1, 0, 2, 2, 0, 0, 0)   # A[:,2:4] x B[2:4,:] -> C[:,0:2]
    add_block(2, 0, 0, 0, 2, 0, 2)   # A[:,0:2] x B[0:2,2:4] -> C[:,2:4]
    add_block(3, 0, 2, 2, 2, 0, 2)   # A[:,2:4] x B[2:4,2:4] -> C[:,2:4]

    # Verify correctness
    U_arr = U
    V_arr = V
    W_arr = W

    for a_row in range(n):
        for a_col in range(m):
            for b_row in range(m):
                for b_col in range(p):
                    # A[a_row, a_col] * B[b_row, b_col] contributes to C[a_row, b_col]
                    # only if a_col == b_row
                    expected = 1.0 if (a_col == b_row) else 0.0
                    a_idx = a_row * m + a_col
                    b_idx = b_row * p + b_col
                    c_idx = b_col * n + a_row
                    actual = np.sum(U_arr[a_idx, :] * V_arr[b_idx, :] * W_arr[c_idx, :])
                    if abs(actual - expected) > 1e-9:
                        # Fallback: return trivial solution
                        U_triv = np.zeros((n*m, n*m*p))
                        V_triv = np.zeros((m*p, n*m*p))
                        W_triv = np.zeros((p*n, n*m*p))
                        for ar in range(n):
                            for ac in range(m):
                                for bc in range(p):
                                    idx = ar * (m*p) + ac * p + bc
                                    U_triv[ar * m + ac, idx] = 1
                                    V_triv[ac * p + bc, idx] = 1
                                    W_triv[bc * n + ar, idx] = 1
                        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
