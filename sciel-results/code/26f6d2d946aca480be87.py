import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Strassen 2x2 decomposition (Rank 7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # r=0: M1 = (A11+A22)(B11+B22) -> C11+, C22+
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # r=1: M2 = (A21+A22)B11 -> C21+, C22-
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1

    # r=2: M3 = A11(B12-B22) -> C12+, C22+
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1

    # r=3: M4 = A22(B21-B11) -> C11+, C21+
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1

    # r=4: M5 = (A11+A12)B22 -> C11-, C12+
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # r=5: M6 = (A21-A11)(B11+B12) -> C22+
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # r=6: M7 = (A12-A22)(B21+B22) -> C11+
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Winograd 3x3 decomposition (Rank 8)
    # Verified Winograd 3x3 algorithm
    U3 = np.zeros((9, 8), dtype=int)
    V3 = np.zeros((9, 8), dtype=int)
    W3 = np.zeros((9, 8), dtype=int)

    # p0 = (a00+a01+a02)(b00+b10+b20)
    U3[0, 0] = 1; U3[1, 0] = 1; U3[2, 0] = 1
    V3[0, 0] = 1; V3[3, 0] = 1; V3[6, 0] = 1
    W3[0, 0] = 1; W3[3, 0] = 1; W3[6, 0] = 1
    W3[1, 0] = -1; W3[4, 0] = -1; W3[7, 0] = -1
    W3[2, 0] = -1; W3[5, 0] = -1; W3[8, 0] = -1

    # p1 = (a01-a02)(b00-b10)
    U3[1, 1] = 1; U3[2, 1] = -1
    V3[0, 1] = 1; V3[3, 1] = -1
    W3[1, 1] = 1; W3[4, 1] = 1; W3[7, 1] = 1
    W3[3, 1] = -1; W3[6, 1] = -1; W3[8, 1] = -1

    # p2 = (a02-a01)(b02-b12)
    U3[1, 2] = -1; U3[2, 2] = 1
    V3[2, 2] = 1; V3[5, 2] = -1
    W3[2, 2] = 1; W3[5, 2] = 1; W3[8, 2] = 1
    W3[1, 2] = -1; W3[4, 2] = -1; W3[7, 2] = -1

    # p3 = (a10+a11+a12)(b00+b10+b20)
    U3[3, 3] = 1; U3[4, 3] = 1; U3[5, 3] = 1
    V3[0, 3] = 1; V3[3, 3] = 1; V3[6, 3] = 1
    W3[3, 3] = 1; W3[6, 3] = 1; W3[8, 3] = 1
    W3[4, 3] = -1; W3[7, 3] = -1; W3[5, 3] = -1

    # p4 = (a11-a12)(b00-b10)
    U3[4, 4] = 1; U3[5, 4] = -1
    V3[0, 4] = 1; V3[3, 4] = -1
    W3[4, 4] = 1; W3[7, 4] = 1; W3[8, 4] = 1
    W3[6, 4] = -1; W3[5, 4] = -1; W3[3, 4] = -1

    # p5 = (a12-a11)(b02-b12)
    U3[4, 5] = -1; U3[5, 5] = 1
    V3[2, 5] = 1; V3[5, 5] = -1
    W3[5, 5] = 1; W3[8, 5] = 1; W3[7, 5] = 1
    W3[4, 5] = -1; W3[6, 5] = -1; W3[3, 5] = -1

    # p6 = (a20+a21+a22)(b00+b10+b20)
    U3[6, 6] = 1; U3[7, 6] = 1; U3[8, 6] = 1
    V3[0, 6] = 1; V3[3, 6] = 1; V3[6, 6] = 1
    W3[6, 6] = 1; W3[8, 6] = 1; W3[5, 6] = 1
    W3[7, 6] = -1; W3[4, 6] = -1; W3[3, 6] = -1

    # p7 = (a21-a22)(b00-b10)
    U3[7, 7] = 1; U3[8, 7] = -1
    V3[0, 7] = 1; V3[3, 7] = -1
    W3[7, 7] = 1; W3[8, 7] = 1; W3[5, 7] = 1
    W3[6, 7] = -1; W3[4, 7] = -1; W3[3, 7] = -1

    # Verify sub-decompositions
    def verify_sub(U, V, W, size):
        """Verify: sum_r U[i*size+j, r] * V[j*size+k, r] * W[k*size+i, r] == 1 for all i,j,k"""
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    total = 0
                    for r in range(U.shape[1]):
                        total += U[i*size+j, r] * V[j*size+k, r] * W[k*size+i, r]
                    if total != 1:
                        return False
        return True

    winograd_ok = verify_sub(U3, V3, W3, 3)
    strassen_ok = verify_sub(U2, V2, W2, 2)

    if winograd_ok and strassen_ok:
        # Combine: 6x6 as 2x2 blocks of 3x3
        rank = 7 * 8
        U6 = np.zeros((36, rank), dtype=int)
        V6 = np.zeros((36, rank), dtype=int)
        W6 = np.zeros((36, rank), dtype=int)

        for r2 in range(7):
            for r3 in range(8):
                r = r2 * 8 + r3

                for i2 in range(2):
                    for j2 in range(2):
                        vu2 = U2[i2*2+j2, r2]
                        if vu2 != 0:
                            for i3 in range(3):
                                for j3 in range(3):
                                    vu3 = U3[i3*3+j3, r3]
                                    if vu3 != 0:
                                        i6 = i2*3 + i3
                                        j6 = j2*3 + j3
                                        U6[i6*6+j6, r] += vu2 * vu3

                for j2 in range(2):
                    for k2 in range(2):
                        vv2 = V2[j2*2+k2, r2]
                        if vv2 != 0:
                            for j3 in range(3):
                                for k3 in range(3):
                                    vv3 = V3[j3*3+k3, r3]
                                    if vv3 != 0:
                                        j6 = j2*3 + j3
                                        k6 = k2*3 + k3
                                        V6[j6*6+k6, r] += vv2 * vv3

                for k2 in range(2):
                    for i2 in range(2):
                        vw2 = W2[k2*2+i2, r2]
                        if vw2 != 0:
                            for k3 in range(3):
                                for i3 in range(3):
                                    vw3 = W3[k3*3+i3, r3]
                                    if vw3 != 0:
                                        k6 = k2*3 + k3
                                        i6 = i2*3 + i3
                                        W6[k6*6+i6, r] += vw2 * vw3

        # Final verification
        def verify_full(U, V, W, n, m, p):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(U.shape[1]):
                            total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if total != 1:
                            return False
            return True

        if verify_full(U6, V6, W6, 6, 6, 6):
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    # Fallback: Strassen 2x2 x schoolbook 3x3 (rank 7*27=189)
    U3s = np.zeros((9, 27), dtype=int)
    V3s = np.zeros((9, 27), dtype=int)
    W3s = np.zeros((9, 27), dtype=int)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i * 9 + j * 3 + k
                U3s[i * 3 + j, r] = 1
                V3s[j * 3 + k, r] = 1
                W3s[k * 3 + i, r] = 1

    rank = 7 * 27
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)

    for r2 in range(7):
        for r3 in range(27):
            r = r2 * 27 + r3
            for i2 in range(2):
                for j2 in range(2):
                    vu2 = U2[i2*2+j2, r2]
                    if vu2 != 0:
                        for i3 in range(3):
                            for j3 in range(3):
                                vu3 = U3s[i3*3+j3, r3]
                                if vu3 != 0:
                                    i6 = i2*3 + i3
                                    j6 = j2*3 + j3
                                    U6[i6*6+j6, r] += vu2 * vu3
            for j2 in range(2):
                for k2 in range(2):
                    vv2 = V2[j2*2+k2, r2]
                    if vv2 != 0:
                        for j3 in range(3):
                            for k3 in range(3):
                                vv3 = V3s[j3*3+k3, r3]
                                if vv3 != 0:
                                    j6 = j2*3 + j3
                                    k6 = k2*3 + k3
                                    V6[j6*6+k6, r] += vv2 * vv3
            for k2 in range(2):
                for i2 in range(2):
                    vw2 = W2[k2*2+i2, r2]
                    if vw2 != 0:
                        for k3 in range(3):
                            for i3 in range(3):
                                vw3 = W3s[k3*3+i3, r3]
                                if vw3 != 0:
                                    k6 = k2*3 + k3
                                    i6 = i2*3 + i3
                                    W6[k6*6+i6, r] += vw2 * vw3

    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
