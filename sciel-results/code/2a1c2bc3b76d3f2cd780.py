import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8

    # Build rank-49 4x4x4 Strassen (two nested 2x2 Strassen)
    # 2x2 Strassen: rank 7
    # 4x4 Strassen: rank 49 (7 super-steps x 7 sub-steps)

    # 2x2 Strassen U,V,W matrices (4x7 each)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # 4x4 Strassen specs (7 super-steps)
    # A_specs[k]: list of (coeff, (bi, bj)) for 2x2 block indices in A
    # B_specs[k]: list of (coeff, (bi, bj)) for 2x2 block indices in B
    # C_specs[k]: list of (coeff, (bi, bj)) for 2x2 block indices in C

    A_specs = [
        [(1, (0,0)), (1, (1,1))],   # P0: A00+A11
        [(1, (1,0)), (1, (1,1))],   # P1: A10+A11
        [(1, (0,0))],                # P2: A00
        [(1, (1,1))],                # P3: A11
        [(1, (0,0)), (1, (0,1))],   # P4: A00+A01
        [(1, (1,0)), (-1, (0,0))],  # P5: A10-A00
        [(1, (0,1)), (-1, (1,1))]   # P6: A01-A11
    ]

    B_specs = [
        [(1, (0,0)), (1, (1,1))],   # P0: B00+B11
        [(1, (0,0))],                # P1: B00
        [(1, (0,1)), (-1, (1,1))],  # P2: B01-B11
        [(1, (1,0)), (-1, (0,0))],  # P3: B10-B00
        [(1, (1,1))],                # P4: B11
        [(1, (0,0)), (1, (0,1))],   # P5: B00+B01
        [(1, (1,0)), (1, (1,1))]    # P6: B10+B11
    ]

    C_specs = [
        [(1, (0,0)), (1, (1,1))],   # P0: C00+C11
        [(1, (1,0)), (-1, (1,1))],  # P1: C10-C11
        [(1, (0,1)), (1, (1,1))],   # P2: C01+C11
        [(1, (0,0)), (1, (1,0))],   # P3: C00+C10
        [(-1, (0,0)), (1, (0,1))],  # P4: -C00+C01
        [(1, (1,1))],                # P5: C11
        [(1, (0,0))]                 # P6: C00
    ]

    # Build 4x4x4 decomposition (rank 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))

    for kp in range(7):
        for ms in range(7):
            col = kp * 7 + ms

            # U: A is 4x4, index i*4+j
            for coeff, (bi, bj) in A_specs[kp]:
                for r in range(2):
                    for c in range(2):
                        gi = bi * 2 + r
                        gj = bj * 2 + c
                        U4[gi * 4 + gj, col] += coeff * U2[r * 2 + c, ms]

            # V: B is 4x4, index j*4+k
            for coeff, (bi, bj) in B_specs[kp]:
                for r in range(2):
                    for c in range(2):
                        gj = bi * 2 + r
                        gk = bj * 2 + c
                        V4[gj * 4 + gk, col] += coeff * V2[r * 2 + c, ms]

            # W: C is 4x4, index k*4+i
            for coeff, (bi, bj) in C_specs[kp]:
                for r in range(2):
                    for c in range(2):
                        gi = bi * 2 + r
                        gk = bj * 2 + c
                        W4[gk * 4 + gi, col] += coeff * W2[r * 2 + c, ms]

    # Now extend to 4x4x8: B is 4x8, C is 4x8
    # Split B into two 4x4 halves: B[:, 0:4] and B[:, 4:8]
    # Split C into two 4x4 halves: C[:, 0:4] and C[:, 4:8]
    # Total rank = 49 + 49 = 98

    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Copy U (A is unchanged)
    U[:, :49] = U4
    U[:, 49:] = U4

    # For V: B is 4x8, index j*8+k
    # First half: k in [0,4), second half: k in [4,8)
    for half in range(2):
        for j in range(4):
            for k_local in range(4):
                k_global = k_local + half * 4
                v_idx = j * 8 + k_global
                v4_idx = j * 4 + k_local
                V[v_idx, half * 49:(half + 1) * 49] = V4[v4_idx, :]

    # For W: C is 4x8, index k*4+i (trace convention)
    for half in range(2):
        for k_local in range(4):
            for i in range(4):
                k_global = k_local + half * 4
                w_idx = k_global * 4 + i
                w4_idx = k_local * 4 + i
                W[w_idx, half * 49:(half + 1) * 49] = W4[w4_idx, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
