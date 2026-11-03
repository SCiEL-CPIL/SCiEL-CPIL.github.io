import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = 6, 6, 6

    # --- Build rank-7 Strassen-like decomposition for 2x2 ---
    # Term (i-row pattern, k-row pattern, j-row pattern) as 2-bit strings
    # We use the standard 7-term Strassen construction.
    # Each term t has:
    #   U2[i,k] = 1 for certain (i,k)
    #   V2[k,j] = 1 for certain (k,j)
    #   W2[j,i] = 1 for certain (j,i)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # Standard Strassen 2x2:
    # M1 = (A11)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11+A12)(B22)
    # M4 = (A22)(B12+B21)
    # M5 = (A11-A22)(B21+B22)
    # M6 = (A12-A22)(B11+B12)
    # M7 = (A21-A11)(B22+B21)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U2[i*2+k, t]:
    U2[0*2+0, 0] = 1  # M1: A11
    U2[1*2+1, 0] = 1  # M1: B11+B22 (k=1 for B22? no)
    # Let me be more careful with indices.
    # A[i,k] where i in {0,1}, k in {0,1}
    # B[k,j] where k in {0,1}, j in {0,1}
    # C[i,j] where i in {0,1}, j in {0,1}
    #
    # M1 = A[0,0] * (B[0,0] + B[1,1])
    U2[0*2+0, 0] = 1
    V2[0*2+0, 0] = 1
    V2[1*2+1, 0] = 1
    W2[0*2+0, 0] = 1
    W2[1*2+1, 0] = 1

    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    U2[1*2+0, 1] = 1
    U2[1*2+1, 1] = 1
    V2[0*2+0, 1] = 1
    W2[1*2+0, 1] = 1
    W2[1*2+1, 1] = -1  # C21 = M2 + M4, C22 = M1 - M2 + M3 + M6

    # M3 = (A[0,0] + A[0,1]) * B[1,1]
    U2[0*2+0, 2] = 1
    U2[0*2+1, 2] = 1
    V2[1*2+1, 2] = 1
    W2[0*2+1, 2] = 1
    W2[1*2+1, 2] = 1

    # M4 = A[1,1] * (B[0,1] + B[1,0])
    U2[1*2+1, 3] = 1
    V2[0*2+1, 3] = 1
    V2[1*2+0, 3] = 1
    W2[0*2+0, 3] = 1
    W2[1*2+0, 3] = 1

    # M5 = (A[0,0] - A[1,1]) * (B[1,0] + B[1,1])
    U2[0*2+0, 4] = 1
    U2[1*2+1, 4] = -1
    V2[1*2+0, 4] = 1
    V2[1*2+1, 4] = 1
    W2[0*2+0, 4] = -1
    W2[0*2+1, 4] = 1
    W2[1*2+1, 4] = 1

    # M6 = (A[0,1] - A[1,1]) * (B[0,0] + B[0,1])
    U2[0*2+1, 5] = 1
    U2[1*2+1, 5] = -1
    V2[0*2+0, 5] = 1
    V2[0*2+1, 5] = 1
    W2[1*2+1, 5] = 1

    # M7 = (A[1,0] - A[0,0]) * (B[1,1] + B[1,0])
    U2[1*2+0, 6] = 1
    U2[0*2+0, 6] = -1
    V2[1*2+1, 6] = 1
    V2[1*2+0, 6] = 1
    W2[0*2+0, 6] = 1
    W2[1*2+1, 6] = -1

    # --- Build rank-9 schoolbook for 3x3 ---
    U3 = np.zeros((9, 9), dtype=int)
    V3 = np.zeros((9, 9), dtype=int)
    W3 = np.zeros((9, 9), dtype=int)
    for i3 in range(3):
        for k3 in range(3):
            t3 = i3 * 3 + k3
            U3[i3 * 3 + k3, t3] = 1
            for j3 in range(3):
                V3[k3 * 3 + j3, t3] = 1
                W3[j3 * 3 + i3, t3] = 1

    # --- Kronecker product: MM_6 = MM_2 ⊗ MM_3 ---
    rank6 = 7 * 9  # 63
    U6 = np.zeros((36, rank6), dtype=int)
    V6 = np.zeros((36, rank6), dtype=int)
    W6 = np.zeros((36, rank6), dtype=int)

    for t2 in range(7):
        for t3 in range(9):
            t = t2 * 9 + t3
            u2_col = U2[:, t2]
            v2_col = V2[:, t2]
            w2_col = W2[:, t2]
            u3_col = U3[:, t3]
            v3_col = V3[:, t3]
            w3_col = W3[:, t3]

            for i2 in range(2):
                for i3 in range(3):
                    for k2 in range(2):
                        for k3 in range(3):
                            u_row = (2 * i2 + i3) * 6 + (2 * k2 + k3)
                            U6[u_row, t] += u2_col[i2 * 2 + k2] * u3_col[i3 * 3 + k3]

            for k2 in range(2):
                for k3 in range(3):
                    for j2 in range(2):
                        for j3 in range(3):
                            v_row = (2 * k2 + k3) * 6 + (2 * j2 + j3)
                            V6[v_row, t] += v2_col[k2 * 2 + j2] * v3_col[k3 * 3 + j3]

            for j2 in range(2):
                for j3 in range(3):
                    for i2 in range(2):
                        for i3 in range(3):
                            w_row = (2 * j2 + j3) * 6 + (2 * i2 + i3)
                            W6[w_row, t] += w2_col[j2 * 2 + i2] * w3_col[j3 * 3 + i3]

    # Verify
    verified = True
    for i in range(6):
        for j in range(6):
            for k in range(6):
                u_row = i * 6 + k
                v_row = k * 6 + j
                w_row = j * 6 + i
                actual = np.sum(U6[u_row, :] * V6[v_row, :] * W6[w_row, :])
                if abs(actual - 1.0) > 1e-10:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break

    if not verified:
        # Fallback: rank-36 Kronecker with schoolbook 2x2
        U2b = np.zeros((4, 4), dtype=int)
        V2b = np.zeros((4, 4), dtype=int)
        W2b = np.zeros((4, 4), dtype=int)
        for i2 in range(2):
            for k2 in range(2):
                t2 = i2 * 2 + k2
                U2b[i2 * 2 + k2, t2] = 1
                for j2 in range(2):
                    V2b[k2 * 2 + j2, t2] = 1
                    W2b[j2 * 2 + i2, t2] = 1

        rank6b = 4 * 9
        U6b = np.zeros((36, rank6b), dtype=int)
        V6b = np.zeros((36, rank6b), dtype=int)
        W6b = np.zeros((36, rank6b), dtype=int)

        for t2 in range(4):
            for t3 in range(9):
                t = t2 * 9 + t3
                u2_col = U2b[:, t2]
                v2_col = V2b[:, t2]
                w2_col = W2b[:, t2]
                u3_col = U3[:, t3]
                v3_col = V3[:, t3]
                w3_col = W3[:, t3]

                for i2 in range(2):
                    for i3 in range(3):
                        for k2 in range(2):
                            for k3 in range(3):
                                u_row = (2 * i2 + i3) * 6 + (2 * k2 + k3)
                                U6b[u_row, t] += u2_col[i2 * 2 + k2] * u3_col[i3 * 3 + k3]

                for k2 in range(2):
                    for k3 in range(3):
                        for j2 in range(2):
                            for j3 in range(3):
                                v_row = (2 * k2 + k3) * 6 + (2 * j2 + j3)
                                V6b[v_row, t] += v2_col[k2 * 2 + j2] * v3_col[k3 * 3 + j3]

                for j2 in range(2):
                    for j3 in range(3):
                        for i2 in range(2):
                            for i3 in range(3):
                                w_row = (2 * j2 + j3) * 6 + (2 * i2 + i3)
                                W6b[w_row, t] += w2_col[j2 * 2 + i2] * w3_col[j3 * 3 + i3]

        return {"U": U6b.tolist(), "V": V6b.tolist(), "W": W6b.tolist()}

    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
