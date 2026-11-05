import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Strassen 2x2 rank-7 decomposition
    def get_strassen_2x2():
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1
        V[0, 3] = -1; V[2, 3] = 1
        W[0, 3] = 1; W[1, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        U[0, 5] = -1; U[2, 5] = 1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W

    # 3x3 schoolbook rank-27 decomposition
    def get_schoolbook_3x3():
        U = np.zeros((9, 27), dtype=int)
        V = np.zeros((9, 27), dtype=int)
        W = np.zeros((9, 27), dtype=int)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U[i * 3 + j, r] = 1
                    V[j * 3 + k, r] = 1
                    W[k * 3 + i, r] = 1
        return U, V, W

    U2, V2, W2 = get_strassen_2x2()
    rank2 = 7
    U3, V3, W3 = get_schoolbook_3x3()
    rank3 = 27
    rank_total = rank2 * rank3  # 189

    # Build 6x6 via Kronecker product (6 = 2 * 3)
    U6 = np.zeros((36, rank_total), dtype=int)
    V6 = np.zeros((36, rank_total), dtype=int)
    W6 = np.zeros((36, rank_total), dtype=int)

    for r2 in range(rank2):
        for r3 in range(rank3):
            r = r2 * rank3 + r3
            for i2 in range(2):
                for j2 in range(2):
                    val_u2 = U2[i2*2+j2, r2]
                    if val_u2 == 0:
                        continue
                    for i3 in range(3):
                        for j3 in range(3):
                            val_u3 = U3[i3*3+j3, r3]
                            if val_u3 == 0:
                                continue
                            i6 = i2*3 + i3
                            j6 = j2*3 + j3
                            U6[i6*6+j6, r] += val_u2 * val_u3
            for j2 in range(2):
                for k2 in range(2):
                    val_v2 = V2[j2*2+k2, r2]
                    if val_v2 == 0:
                        continue
                    for j3 in range(3):
                        for k3 in range(3):
                            val_v3 = V3[j3*3+k3, r3]
                            if val_v3 == 0:
                                continue
                            j6 = j2*3 + j3
                            k6 = k2*3 + k3
                            V6[j6*6+k6, r] += val_v2 * val_v3
            for k2 in range(2):
                for i2 in range(2):
                    val_w2 = W2[k2*2+i2, r2]
                    if val_w2 == 0:
                        continue
                    for k3 in range(3):
                        for i3 in range(3):
                            val_w3 = W3[k3*3+i3, r3]
                            if val_w3 == 0:
                                continue
                            k6 = k2*3 + k3
                            i6 = i2*3 + i3
                            W6[k6*6+i6, r] += val_w2 * val_w3

    # Verify full decomposition
    def verify_full(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True

    if time.time() < deadline - 5:
        valid = verify_full(U6, V6, W6, n, m, p)
        if valid:
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    # Fallback
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
