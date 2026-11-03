import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 4, 4, 8

    # Strassen 2x2x2 factors (rank 7)
    U2 = np.zeros((2, 2, 7), dtype=float)
    V2 = np.zeros((2, 2, 7), dtype=float)
    W2 = np.zeros((2, 2, 7), dtype=float)

    # r=0: (A00+A11)(B00+B11)
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1

    # r=1: (A10+A11)B00
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1

    # r=2: A00(B01-B11)
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1

    # r=3: A11(B10-B00)
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1

    # r=4: (A00+A01)B11
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1

    # r=5: (A10-A00)(B00+B01)
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1

    # r=6: (A01-A11)(B10+B11)
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1

    # Build 4x4x4 factors using Kronecker product of Strassen (rank 49)
    U4 = np.zeros((4, 4, 49), dtype=float)
    V4 = np.zeros((4, 4, 49), dtype=float)
    W4 = np.zeros((4, 4, 49), dtype=float)

    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2 * i1 + i0
                    j = 2 * j1 + j0
                    for s in range(7):
                        for r in range(7):
                            R = 7 * s + r
                            U4[i, j, R] = U2[i1, j1, s] * U2[i0, j0, r]

    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2 * j1 + j0
                    k = 2 * k1 + k0
                    for s in range(7):
                        for r in range(7):
                            R = 7 * s + r
                            V4[j, k, R] = V2[j1, k1, s] * V2[j0, k0, r]

    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2 * k1 + k0
                    i = 2 * i1 + i0
                    for s in range(7):
                        for r in range(7):
                            R = 7 * s + r
                            W4[k, i, R] = W2[k1, i1, s] * W2[k0, i0, r]

    # Build 4x4x8 factors
    # Split B (4x8) into two 4x4 blocks: B[:, 0:4] and B[:, 4:8]
    # C (4x8) split into C[:, 0:4] and C[:, 4:8]
    # Each half is a 4x4x4 multiplication
    # Total rank = 2 * 49 = 98

    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)

    # First half: k in 0..3
    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i * 4 + j, R] = U4[i, j, R]

    for j in range(4):
        for k in range(4):
            for R in range(49):
                V[j * 8 + k, R] = V4[j, k, R]

    for k in range(4):
        for i in range(4):
            for R in range(49):
                W[k * 4 + i, R] = W4[k, i, R]

    # Second half: k in 4..7, mapped to k' = k-4 in 0..3
    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i * 4 + j, 49 + R] = U4[i, j, R]

    for j in range(4):
        for kp in range(4):
            k = 4 + kp
            for R in range(49):
                V[j * 8 + k, 49 + R] = V4[j, kp, R]

    for kp in range(4):
        k = 4 + kp
        for i in range(4):
            for R in range(49):
                W[k * 4 + i, 49 + R] = W4[kp, i, R]

    # Verification
    if time.time() < deadline - 5.0:
        U3 = U.reshape(4, 4, 98)
        V3 = V.reshape(4, 8, 98)
        W3 = W.reshape(8, 4, 98)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        if not np.allclose(result, 1.0, atol=1e-10):
            # Fallback to schoolbook
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school))
            V_s = np.zeros((m * p, rank_school))
            W_s = np.zeros((p * n, rank_school))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j, col] = 1
                        V_s[j * p + k, col] = 1
                        W_s[k * n + i, col] = 1
                        col += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
