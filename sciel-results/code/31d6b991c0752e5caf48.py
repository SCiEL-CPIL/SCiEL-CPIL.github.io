import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = 4, 4, 6

    # Rank 40 decomposition:
    # Part 1: Strassen for 4x4 x 4x4 (first 4 columns of B) -> rank 12
    # Part 2: 4 Strassen for 2x2 x 2x2 (last 2 columns of B) -> rank 28
    # Total rank: 40

    # Strassen 2x2x2 coefficients
    # alpha: 7 combinations of A blocks (each A block is 2x2, indexed 0-3)
    # beta: 7 combinations of B blocks
    # recon: reconstruction to get C blocks

    alpha_2x2 = [
        [1, 0, 0, 1],   # s0: A00+A11
        [0, 0, 1, 1],   # s1: A10+A11
        [1, 0, 0, 0],   # s2: A00
        [0, 0, 0, 1],   # s3: A11
        [1, 1, 0, 0],   # s4: A00+A01
        [-1, 0, 1, 0],  # s5: A10-A00
        [0, 1, 0, -1]   # s6: A01-A11
    ]
    beta_2x2 = [
        [1, 0, 0, 1],   # s0: B00+B11
        [1, 0, 0, 0],   # s1: B00
        [0, 1, 0, -1],  # s2: B01-B11
        [-1, 0, 1, 0],  # s3: B10-B00
        [0, 0, 0, 1],   # s4: B11
        [1, 1, 0, 0],   # s5: B00+B01
        [0, 0, 1, 1]    # s6: B10+B11
    ]
    # recon[s] = list of (a_block, b_block, coeff) for C blocks
    # C00, C10, C01, C11
    recon_2x2 = [
        [(0, 0, 1), (1, 1, 1)],     # C00
        [(1, 0, 1), (1, 1, -1)],    # C10
        [(0, 1, 1), (1, 1, 1)],     # C01
        [(0, 0, 1), (1, 0, 1)],     # C11
        [(0, 0, -1), (0, 1, 1)],    # C00
        [(1, 1, 1)],                # C11
        [(0, 0, 1)]                 # C00
    ]

    R = 40
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))

    # Helper: index for A (4x4), flat row-major
    def a_idx(r, c):
        return r * 4 + c

    # Helper: index for B (4x6), flat row-major
    def b_idx(r, c):
        return r * 6 + c

    # Helper: index for C (4x6), flat row-major
    def c_idx(r, c):
        return r * 6 + c

    col = 0

    # Part 1: Strassen for 4x4 x 4x4 (columns 0-3 of B)
    # Split A into 2x2 blocks of 2x2: A00, A01, A10, A11
    # Split B into 2x2 blocks of 2x2: B00, B01, B10, B11 (each 2x2, using cols 0-3)
    # C = A*B, split into 2x2 blocks of 2x2: C00, C01, C10, C11

    # Top-level Strassen on 4x4 blocks
    alpha_top = [
        [1, 0, 0, 1],   # S0: A00+A11
        [0, 0, 1, 1],   # S1: A10+A11
        [1, 0, 0, 0],   # S2: A00
        [0, 0, 0, 1],   # S3: A11
        [1, 1, 0, 0],   # S4: A00+A01
        [-1, 0, 1, 0],  # S5: A10-A00
        [0, 1, 0, -1]   # S6: A01-A11
    ]
    beta_top = [
        [1, 0, 0, 1],   # S0: B00+B11
        [1, 0, 0, 0],   # S1: B00
        [0, 1, 0, -1],  # S2: B01-B11
        [-1, 0, 1, 0],  # S3: B10-B00
        [0, 0, 0, 1],   # S4: B11
        [1, 1, 0, 0],   # S5: B00+B01
        [0, 0, 1, 1]    # S6: B10+B11
    ]
    recon_top = [
        [(0, 0, 1), (1, 1, 1)],     # C00
        [(1, 0, 1), (1, 1, -1)],    # C10
        [(0, 1, 1), (1, 1, 1)],     # C01
        [(0, 0, 1), (1, 0, 1)],     # C11
        [(0, 0, -1), (0, 1, 1)],    # C00
        [(1, 1, 1)],                # C11
        [(0, 0, 1)]                 # C00
    ]

    # For each top-level Strassen term s, we need to compute
    # (sum of A blocks) * (sum of B blocks) where each is 2x2 x 2x2
    # Use inner Strassen for each 2x2x2x2 product

    for s in range(7):
        if time.time() > deadline:
            break

        # U: combine top-level alpha[s] with inner alpha_2x2[si]
        for si in range(7):
            # U contribution
            for a_block in range(4):
                if alpha_2x2[si][a_block] == 0:
                    continue
                ar, ac = a_block // 2, a_block % 2
                for top_a in range(4):
                    if alpha_top[s][top_a] == 0:
                        continue
                    tar, tac = top_a // 2, top_a % 2
                    # A element at (tar*2+ar, tac*2+ac)
                    u_idx = a_idx(tar * 2 + ar, tac * 2 + ac)
                    U[u_idx, col] += alpha_2x2[si][a_block] * alpha_top[s][top_a]

            # V contribution
            for b_block in range(4):
                if beta_2x2[si][b_block] == 0:
                    continue
                br, bc = b_block // 2, b_block % 2
                for top_b in range(4):
                    if beta_top[s][top_b] == 0:
                        continue
                    tar, tac = top_b // 2, top_b % 2
                    # B element at (tar*2+br, tac*2+bc)
                    v_idx = b_idx(tar * 2 + br, tac * 2 + bc)
                    V[v_idx, col] += beta_2x2[si][b_block] * beta_top[s][top_b]

            # W contribution
            for (ca, cb, gamma) in recon_top[s]:
                for (za, zb, zgamma) in recon_2x2[si]:
                    # C block (ca, cb) gets contribution from inner C block (za, zb)
                    w_idx = c_idx(ca * 2 + za, cb * 2 + zb)
                    W[w_idx, col] += gamma * zgamma

            col += 1

    # Part 2: 4 Strassen for 2x2 x 2x2 (last 2 columns of B)
    # C[:, 4:6] = A[:, 0:4] * B[:, 4:6]
    # Split into 4 terms:
    # C[0:2, 4:6] = A[0:2, 0:2]*B[0:2, 4:6] + A[0:2, 2:4]*B[2:4, 4:6]
    # C[2:4, 4:6] = A[2:4, 0:2]*B[0:2, 4:6] + A[2:4, 2:4]*B[2:4, 4:6]
    # Each is 2x2 x 2x2, use Strassen

    # Term 1: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    # Term 2: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    # Term 3: A[2:4, 0:2] * B[0:2, 4:6] -> C[2:4, 4:6]
    # Term 4: A[2:4, 2:4] * B[2:4, 4:6] -> C[2:4, 4:6]

    for term in range(4):
        if time.time() > deadline:
            break

        # Determine A row range, A col range, B row range, C row range
        if term == 0:
            a_r0, a_r1 = 0, 2
            a_c0, a_c1 = 0, 2
            b_r0, b_r1 = 0, 2
            c_r0, c_r1 = 0, 2
        elif term == 1:
            a_r0, a_r1 = 0, 2
            a_c0, a_c1 = 2, 4
            b_r0, b_r1 = 2, 4
            c_r0, c_r1 = 0, 2
        elif term == 2:
            a_r0, a_r1 = 2, 4
            a_c0, a_c1 = 0, 2
            b_r0, b_r1 = 0, 2
            c_r0, c_r1 = 2, 4
        else:  # term == 3
            a_r0, a_r1 = 2, 4
            a_c0, a_c1 = 2, 4
            b_r0, b_r1 = 2, 4
            c_r0, c_r1 = 2, 4

        # B columns are always 4:6
        b_c0, b_c1 = 4, 6
        c_c0, c_c1 = 4, 6

        for si in range(7):
            # U contribution
            for a_block in range(4):
                if alpha_2x2[si][a_block] == 0:
                    continue
                ar, ac = a_block // 2, a_block % 2
                u_idx = a_idx(a_r0 + ar, a_c0 + ac)
                U[u_idx, col] += alpha_2x2[si][a_block]

            # V contribution
            for b_block in range(4):
                if beta_2x2[si][b_block] == 0:
                    continue
                br, bc = b_block // 2, b_block % 2
                v_idx = b_idx(b_r0 + br, b_c0 + bc)
                V[v_idx, col] += beta_2x2[si][b_block]

            # W contribution
            for (ca, cb, gamma) in recon_2x2[si]:
                w_idx = c_idx(c_r0 + ca, c_c0 + cb)
                W[w_idx, col] += gamma

            col += 1

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
