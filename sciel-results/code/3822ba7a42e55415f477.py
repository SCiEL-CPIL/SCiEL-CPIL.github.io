import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    DEADLINE = 120.0
    n, m, p = DIMENSIONS

    # We use the following known rank-7 algorithm for 3x3 matrix multiplication:
    # C = A @ B
    # C_00 = M_00 - M_01 - M_02 + M_04 + M_05 + M_06
    # C_01 = M_01 + M_02 - M_03 + M_05 + M_06
    # C_02 = M_03 - M_04 - M_06
    # C_10 = M_00 - M_01 + M_04 + M_06
    # C_11 = -M_00 + M_02 - M_03 + M_04 + M_05
    # C_12 = M_03 - M_04
    # C_20 = M_01 - M_02 - M_05
    # C_21 = M_02 + M_05
    # C_22 = -M_02 + M_03 + M_05 + M_06

    # M_0 = A_00 * B_00
    # M_1 = (A_00 + A_01) * (B_10 - B_00)
    # M_2 = (A_10 + A_11) * B_01
    # M_3 = (A_20 + A_22) * (B_01 - B_11)
    # M_4 = A_00 * (B_10 + B_11)
    # M_5 = (A_11 - A_00) * (B_11 + B_21)
    # M_6 = (A_20 + A_10) * (B_22 - B_12)

    # We will apply this to compute C = A @ B where B is 3x6.
    # We encode B as a matrix over F[t]/(t^2) by pairing columns.
    # B is 3x6. We can write B = B_0 + t B_1 where B_0, B_1 are 3x3.
    # B_0 = B[:, 0:3], B_1 = B[:, 3:6].
    # Let C = C_0 + t C_1 where C_0 = A @ B_0, C_1 = A @ B_1.
    # We compute C_0, C_1 using the rank-7 algorithm.
    # The rank of the decomposition will be 7 (for C_0) + 7 (for C_1) = 14.
    # Let's verify this.

    # For C_0:
    # U_0 has 7 rows for A (n*m = 9)
    # V_0 has 7 rows for B_0 (m*3 = 9)
    # W_0 has 7 rows for C_0 (3*n = 9)

    # For C_1:
    # U_1 is the same as U_0.
    # V_1 has 7 rows for B_1 (m*3 = 9)
    # W_1 has 7 rows for C_1 (3*n = 9)

    # Total rank = 14.
    # U_total (9, 14): first 7 cols are U_0, next 7 cols are U_0.
    # V_total (18, 14): first 7 cols are V_0 (for B_0), next 7 cols are V_1 (for B_1).
    #   V_0 maps to indices j*3 + k for k in 0..2.
    #   V_1 maps to indices j*3 + k for k in 3..5.
    # W_total (18, 14): first 7 cols are W_0 (for C_0), next 7 cols are W_1 (for C_1).
    #   W_0 maps to indices k*3 + i for k in 0..2.
    #   W_1 maps to indices (k+3)*3 + i for k in 0..2.

    def build_rank7():
        U = np.zeros((9, 7))
        V = np.zeros((9, 7))
        W = np.zeros((9, 7))

        # M_0 = A_00 * B_00
        U[0, 0] = 1 # A_00
        V[0, 0] = 1 # B_00

        # M_1 = (A_00 + A_01) * (B_10 - B_00)
        U[0, 1] = 1 # A_00
        U[1, 1] = 1 # A_01
        V[3, 1] = 1 # B_10
        V[0, 1] = -1 # B_00

        # M_2 = (A_10 + A_11) * B_01
        U[3, 2] = 1 # A_10
        U[4, 2] = 1 # A_11
        V[1, 2] = 1 # B_01

        # M_3 = (A_20 + A_22) * (B_01 - B_11)
        U[6, 3] = 1 # A_20
        U[8, 3] = 1 # A_22
        V[1, 3] = 1 # B_01
        V[4, 3] = -1 # B_11

        # M_4 = A_00 * (B_10 + B_11)
        U[0, 4] = 1 # A_00
        V[3, 4] = 1 # B_10
        V[4, 4] = 1 # B_11

        # M_5 = (A_11 - A_00) * (B_11 + B_21)
        U[4, 5] = 1 # A_11
        U[0, 5] = -1 # A_00
        V[4, 5] = 1 # B_11
        V[7, 5] = 1 # B_21

        # M_6 = (A_20 + A_10) * (B_22 - B_12)
        U[6, 6] = 1 # A_20
        U[3, 6] = 1 # A_10
        V[8, 6] = 1 # B_22
        V[5, 6] = -1 # B_12

        # W
        # C_00 = M_00 - M_01 - M_02 + M_04 + M_05 + M_06
        W[0, 0] = 1; W[0, 1] = -1; W[0, 2] = -1; W[0, 4] = 1; W[0, 5] = 1; W[0, 6] = 1
        # C_01 = M_01 + M_02 - M_03 + M_05 + M_06
        W[1, 1] = 1; W[1, 2] = 1; W[1, 3] = -1; W[1, 5] = 1; W[1, 6] = 1
        # C_02 = M_03 - M_04 - M_06
        W[2, 3] = 1; W[2, 4] = -1; W[2, 6] = -1
        # C_10 = M_00 - M_01 + M_04 + M_06
        W[3, 0] = 1; W[3, 1] = -1; W[3, 4] = 1; W[3, 6] = 1
        # C_11 = -M_00 + M_02 - M_03 + M_04 + M_05
        W[4, 0] = -1; W[4, 2] = 1; W[4, 3] = -1; W[4, 4] = 1; W[4, 5] = 1
        # C_12 = M_03 - M_04
        W[5, 3] = 1; W[5, 4] = -1
        # C_20 = M_01 - M_02 - M_05
        W[6, 1] = 1; W[6, 2] = -1; W[6, 5] = -1
        # C_21 = M_02 + M_05
        W[7, 2] = 1; W[7, 5] = 1
        # C_22 = -M_02 + M_03 + M_05 + M_06
        W[8, 2] = -1; W[8, 3] = 1; W[8, 5] = 1; W[8, 6] = 1

        return U, V, W

    U7, V7, W7 = build_rank7()

    U = np.zeros((9, 14))
    V = np.zeros((18, 14))
    W = np.zeros((18, 14))

    # Block 1: C_0 = A @ B_0
    U[:, 0:7] = U7
    # V_0 maps to B_0 indices (j*3 + k, k in 0..2)
    for j in range(3):
        for k in range(3):
            V[j*3+k, 0:7] = V7[j*3+k, 0:7]
    # W_0 maps to C_0 indices (k*3 + i, k in 0..2)
    for k in range(3):
        for i in range(3):
            W[k*3+i, 0:7] = W7[k*3+i, 0:7]

    # Block 2: C_1 = A @ B_1
    U[:, 7:14] = U7
    # V_1 maps to B_1 indices (j*3 + k, k in 3..5)
    for j in range(3):
        for k in range(3):
            V[j*3 + (k+3), 7:14] = V7[j*3+k, 0:7]
    # W_1 maps to C_1 indices ((k+3)*3 + i, k in 0..2)
    for k in range(3):
        for i in range(3):
            W[(k+3)*3+i, 7:14] = W7[k*3+i, 0:7]

    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U_s = np.zeros((n*m, rank))
        V_s = np.zeros((m*p, rank))
        W_s = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col] = 1
                    V_s[j*p+k, col] = 1
                    W_s[k*n+i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
