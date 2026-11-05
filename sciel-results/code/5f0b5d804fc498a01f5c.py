"""Combined: Kronecker product of Laderman 3x3 (rank 23) and Strassen 2x2 (rank 7) for 6x6x6."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s wall-clock limit

    # Laderman's 3x3 matrix multiplication algorithm (rank 23)
    laderman_23_patterns = [
        ({(0,0):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1}),
        ({(1,0):1}, {(1,0):1,(2,0):1}, {(1,0):1}),
        ({(2,0):1}, {(2,0):1}, {(2,0):1}),
        ({(1,1):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,0):1,(2,0):-1}),
        ({(1,1):1}, {(2,1):1}, {(1,0):-1}),
        ({(2,1):1}, {(2,1):1}, {(2,0):1}),
        ({(1,2):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,0):-1,(2,0):1}),
        ({(1,2):1}, {(2,2):1}, {(1,0):1}),
        ({(2,2):1}, {(2,2):1}, {(2,0):-1}),
        ({(0,1):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,1):1}),
        ({(0,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,2):1}),
        ({(1,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1}),
        ({(1,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1}),
        ({(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(2,1):1}),
        ({(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(2,2):1}),
        ({(0,0):1}, {(1,1):1,(2,1):1}, {(0,1):1}),
        ({(0,0):1}, {(1,2):1,(2,2):1}, {(0,2):1}),
        ({(1,1):1}, {(2,2):1}, {(1,2):-1}),
        ({(1,2):1}, {(2,1):1}, {(1,1):-1}),
        ({(2,1):1}, {(2,2):1}, {(2,2):-1}),
        ({(2,2):1}, {(2,1):1}, {(2,1):-1}),
        ({(1,1):1,(2,2):1}, {(0,1):1,(1,2):1}, {(1,1):1,(2,2):1}),
        ({(1,2):1,(2,1):1}, {(0,2):1,(1,1):1}, {(1,2):1,(2,1):1}),
    ]

    # Build Laderman 3x3 factor matrices: U33 (9,23), V33 (9,23), W33 (9,23)
    U33 = np.zeros((9, 23), dtype=float)
    V33 = np.zeros((9, 23), dtype=float)
    W33 = np.zeros((9, 23), dtype=float)

    for r_idx, (a_coeffs, b_coeffs, c_coeffs) in enumerate(laderman_23_patterns):
        for (pi, pj), val in a_coeffs.items():
            U33[pi * 3 + pj, r_idx] += val
        for (pi, pj), val in b_coeffs.items():
            V33[pi * 3 + pj, r_idx] += val
        for (pi, pj), val in c_coeffs.items():
            W33[pj * 3 + pi, r_idx] += val  # W index = k*3+i

    # Build Strassen 2x2 factor matrices: U22 (4,7), V22 (4,7), W22 (4,7)
    U22 = np.zeros((4, 7), dtype=float)
    V22 = np.zeros((4, 7), dtype=float)
    W22 = np.zeros((4, 7), dtype=float)

    # P1: (x00+x11)(y00+y11) -> c00+=1, c11+=1
    U22[0, 0] = 1; U22[3, 0] = 1
    V22[0, 0] = 1; V22[3, 0] = 1
    W22[0, 0] = 1; W22[3, 0] = 1

    # P2: (x10+x11)(y00) -> c10+=1, c11+=-1
    U22[2, 1] = 1; U22[3, 1] = 1
    V22[0, 1] = 1
    W22[1, 1] = 1; W22[3, 1] = -1

    # P3: (x00)(y01-y11) -> c01+=1, c11+=1
    U22[0, 2] = 1
    V22[1, 2] = 1; V22[3, 2] = -1
    W22[2, 2] = 1; W22[3, 2] = 1

    # P4: (x11)(y10-y00) -> c00+=1, c10+=1
    U22[3, 3] = 1
    V22[2, 3] = 1; V22[0, 3] = -1
    W22[0, 3] = 1; W22[1, 3] = 1

    # P5: (x00+x01)(y11) -> c00+=-1, c01+=1
    U22[0, 4] = 1; U22[1, 4] = 1
    V22[3, 4] = 1
    W22[0, 4] = -1; W22[2, 4] = 1

    # P6: (x10-x00)(y00+y01) -> c11+=1
    U22[2, 5] = 1; U22[0, 5] = -1
    V22[0, 5] = 1; V22[1, 5] = 1
    W22[3, 5] = 1

    # P7: (x01-x11)(y10+y11) -> c00+=1
    U22[1, 6] = 1; U22[3, 6] = -1
    V22[2, 6] = 1; V22[3, 6] = 1
    W22[0, 6] = 1

    # Kronecker product: rank = 23 * 7 = 161
    rank = 23 * 7  # 161
    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)

    # For each pair (r33, r22), column c = r33*7 + r22
    # U66 row: (i3*2+i2)*6 + (j3*2+j2)
    # V66 row: (j3*2+j2)*6 + (k3*2+k2)
    # W66 row: (k3*2+k2)*6 + (i3*2+i2)

    for r33 in range(23):
        for r22 in range(7):
            c_final = r33 * 7 + r22

            # U66
            for i3 in range(3):
                for j3 in range(3):
                    u33_val = U33[i3 * 3 + j3, r33]
                    if u33_val == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u22_val = U22[i2 * 2 + j2, r22]
                            if u22_val == 0:
                                continue
                            target_row = (i3 * 2 + i2) * 6 + (j3 * 2 + j2)
                            U66[target_row, c_final] = u33_val * u22_val

            # V66
            for j3 in range(3):
                for k3 in range(3):
                    v33_val = V33[j3 * 3 + k3, r33]
                    if v33_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v22_val = V22[j2 * 2 + k2, r22]
                            if v22_val == 0:
                                continue
                            target_row = (j3 * 2 + j2) * 6 + (k3 * 2 + k2)
                            V66[target_row, c_final] = v33_val * v22_val

            # W66
            for k3 in range(3):
                for i3 in range(3):
                    w33_val = W33[k3 * 3 + i3, r33]
                    if w33_val == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w22_val = W22[k2 * 2 + i2, r22]
                            if w22_val == 0:
                                continue
                            target_row = (k3 * 2 + k2) * 6 + (i3 * 2 + i2)
                            W66[target_row, c_final] = w33_val * w22_val

    # Verify correctness
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, size=(n, m))
        B_test = np.random.randint(-5, 6, size=(m, p))
        C_expected = A_test @ B_test

        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += A_test[i, j] * B_test[j, k] * U66[i*m+j, r] * V66[j*p+k, r] * W66[k*n+i, r]
                C_computed[i, k] = s

        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to Laderman 3x3 blocks (rank 184)
            rank_fb = 8 * 23  # 184
            U_fb = np.zeros((36, rank_fb), dtype=float)
            V_fb = np.zeros((36, rank_fb), dtype=float)
            W_fb = np.zeros((36, rank_fb), dtype=float)

            col_idx = 0
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        for a_coeffs, b_coeffs, c_coeffs in laderman_23_patterns:
                            for (pi, pj), val in a_coeffs.items():
                                r = I * 3 + pi
                                c = J * 3 + pj
                                U_fb[r * 6 + c, col_idx] += val
                            for (pi, pj), val in b_coeffs.items():
                                r = J * 3 + pi
                                c = K * 3 + pj
                                V_fb[r * 6 + c, col_idx] += val
                            for (pi, pj), val in c_coeffs.items():
                                i = I * 3 + pi
                                k = K * 3 + pj
                                W_fb[k * 6 + i, col_idx] += val
                            col_idx += 1

            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}
# EVOLVE-BLOCK-END
