import numpy as np
import time

def construct():
    """
    Returns a rank-98 exact tensor decomposition for 4x4x8 matrix multiplication.
    Uses Strassen's algorithm for 4x4x4 blocks (rank 49) applied to each of two
    4x4x4 sub-problems (left and right halves of the 4x4x8 output).
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin

    n, m, p = 4, 4, 8

    # Strassen 2x2x2 decomposition (rank 7)
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build 4x4x4 decomposition from two-level Strassen (rank 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))

    # Index mapping arrays
    for uo in range(4):
        for ui in range(4):
            I = uo // 2
            J = uo % 2
            ip = ui // 2
            jp = ui % 2
            i = 2 * I + ip
            j = 2 * J + jp
            U_idx = i * 4 + j

            for ro in range(7):
                if U2[uo, ro] != 0:
                    for ri in range(7):
                        if U2[ui, ri] != 0:
                            r = ro * 7 + ri
                            U4[U_idx, r] += U2[uo, ro] * U2[ui, ri]

            for vo in range(4):
                if V2[vo, ro] != 0:
                    for vi in range(4):
                        if V2[vi, ri] != 0:
                            J_v = vo // 2
                            K_v = vo % 2
                            jp_v = vi // 2
                            kp_v = vi % 2
                            j_v = 2 * J_v + jp_v
                            k_v = 2 * K_v + kp_v
                            V_idx = j_v * 4 + k_v
                            r = ro * 7 + ri
                            V4[V_idx, r] += V2[vo, ro] * V2[vi, ri]

            for wo in range(4):
                if W2[wo, ro] != 0:
                    for wi in range(4):
                        if W2[wi, ri] != 0:
                            K_w = wo // 2
                            I_w = wo % 2
                            kp_w = wi // 2
                            ip_w = wi % 2
                            k_w = 2 * K_w + kp_w
                            i_w = 2 * I_w + ip_w
                            W_idx = k_w * 4 + i_w
                            r = ro * 7 + ri
                            W4[W_idx, r] += W2[wo, ro] * W2[wi, ri]

    # Construct 4x4x8 decomposition (Rank 98)
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))

    # Left block (columns 0-3 of B, i.e., k=0..3 in C)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
            W[k * 4 + j, 0:49] = W4[k * 4 + j, :]

    # Right block (columns 4-7 of B, i.e., k=4..7 in C)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V4[j * 4 + k, :]
            W[(4 + k) * 4 + j, 49:98] = W4[k * 4 + j, :]

    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
