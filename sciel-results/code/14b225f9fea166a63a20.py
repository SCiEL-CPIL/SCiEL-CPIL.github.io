"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 4, 8

    # Strassen's 7-term decomposition for 2x2 matrix multiplication
    U2x2 = np.zeros((4, 7))
    V2x2 = np.zeros((4, 7))
    W2x2 = np.zeros((4, 7))

    # M1: (A00 + A11)(B00 + B11)
    U2x2[0, 0] = 1; U2x2[3, 0] = 1
    V2x2[0, 0] = 1; V2x2[3, 0] = 1
    W2x2[0, 0] = 1; W2x2[3, 0] = 1

    # M2: (A10 + A11)B00
    U2x2[2, 1] = 1; U2x2[3, 1] = 1
    V2x2[0, 1] = 1
    W2x2[1, 1] = 1; W2x2[3, 1] = -1

    # M3: A00(B01 - B11)
    U2x2[0, 2] = 1
    V2x2[1, 2] = 1; V2x2[3, 2] = -1
    W2x2[2, 2] = 1; W2x2[3, 2] = 1

    # M4: A11(B10 - B00)
    U2x2[3, 3] = 1
    V2x2[2, 3] = 1; V2x2[0, 3] = -1
    W2x2[0, 3] = 1; W2x2[1, 3] = 1

    # M5: (A00 + A01)B11
    U2x2[0, 4] = 1; U2x2[1, 4] = 1
    V2x2[3, 4] = 1
    W2x2[0, 4] = -1; W2x2[2, 4] = 1

    # M6: (A10 - A00)(B00 + B01)
    U2x2[2, 5] = 1; U2x2[0, 5] = -1
    V2x2[0, 5] = 1; V2x2[1, 5] = 1
    W2x2[3, 5] = 1

    # M7: (A01 - A11)(B10 + B11)
    U2x2[1, 6] = 1; U2x2[3, 6] = -1
    V2x2[2, 6] = 1; V2x2[3, 6] = 1
    W2x2[0, 6] = 1

    # For 2x4 * 4x8:
    # Partition A into two 2x2 blocks: A_left (cols 0,1), A_right (cols 2,3)
    # Partition B into 8 2x2 blocks based on row pairs and column pairs
    # C = sum of 8 (2x2)*(2x2) multiplications

    rank = 56
    U = np.zeros((n*m, rank))  # (8, 56)
    V = np.zeros((m*p, rank))  # (32, 56)
    W = np.zeros((p*n, rank))  # (16, 56)

    for t in range(8):
        if t % 2 == 0:
            a_offset = 0
        else:
            a_offset = 2

        if t < 2:
            b_row_offset = 0 if t == 0 else 2
            c_offset = 0
        elif t < 4:
            b_row_offset = 0 if t == 2 else 2
            c_offset = 2
        elif t < 6:
            b_row_offset = 0 if t == 4 else 2
            c_offset = 4
        else:
            b_row_offset = 0 if t == 6 else 2
            c_offset = 6

        for r in range(7):
            global_r = t * 7 + r

            for i in range(2):
                for k in range(2):
                    local_idx = i * 2 + k
                    val = U2x2[local_idx, r]
                    if val != 0:
                        global_k = k + a_offset
                        U[i * 4 + global_k, global_r] += val

            for k in range(2):
                for j in range(2):
                    local_idx = k * 2 + j
                    val = V2x2[local_idx, r]
                    if val != 0:
                        global_k = k + b_row_offset
                        global_j = j + c_offset
                        V[global_k * 8 + global_j, global_r] += val

            for i in range(2):
                for j in range(2):
                    local_idx = j * 2 + i
                    val = W2x2[local_idx, r]
                    if val != 0:
                        global_j = j + c_offset
                        W[global_j * 2 + i, global_r] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
