"""
Matrix multiplication 4x4x5: rank 65 decomposition via recursive Strassen.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 135.0

    n, m, p = DIMENSIONS  # 4, 4, 5
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20

    R1 = 49  # rank for 4x4x4 via recursive Strassen
    R2 = 16  # rank for 4x4x1 via schoolbook
    R = R1 + R2  # 65

    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)

    # Schoolbook for the 5th column (k=4) of B
    for idx in range(R2):
        i = idx // m
        j = idx % m
        col = R1 + idx
        U[i * m + j, col] = 1.0
        V[j * p + 4, col] = 1.0
        W[4 * n + i, col] = 1.0

    # Strassen coefficients for 2x2x2
    # 7 products, each with A-side, B-side, and C-side contributions
    # Block indices: 0=TL, 1=TR, 2=BL, 3=BR (within a 2x2 block)
    
    # A-side: list of (block_idx, coeff) for each of the 7 products
    SA = [
        [(0, 1), (3, 1)],    # M1: A00+A11
        [(2, 1), (3, 1)],    # M2: A10+A11
        [(0, 1)],             # M3: A00
        [(3, 1)],             # M4: A11
        [(0, 1), (1, 1)],    # M5: A00+A01
        [(2, 1), (0, -1)],   # M6: A10-A00
        [(1, 1), (3, -1)],   # M7: A01-A11
    ]

    # B-side
    SB = [
        [(0, 1), (3, 1)],    # M1: B00+B11
        [(0, 1)],             # M2: B00
        [(1, 1), (3, -1)],   # M3: B01-B11
        [(2, 1), (0, -1)],   # M4: B10-B00
        [(3, 1)],             # M5: B11
        [(0, 1), (1, 1)],    # M6: B00+B01
        [(2, 1), (3, 1)],    # M7: B10+B11
    ]

    # C-side: dict {block_idx: coeff} for each of the 7 products
    SC = [
        {0: 1, 3: 1},       # M1 -> C00 += 1, C11 += 1
        {2: 1},              # M2 -> C10 += 1
        {1: 1, 3: -1},      # M3 -> C01 += 1, C11 -= 1
        {0: -1, 2: 1},      # M4 -> C00 -= 1, C10 += 1
        {1: 1, 3: 1},       # M5 -> C01 += 1, C11 += 1
        {0: 1, 3: -1},      # M6 -> C00 += 1, C11 -= 1
        {0: 1, 3: 1},       # M7 -> C00 += 1, C11 += 1
    ]

    def a_pos(ob, ib):
        """Position (row, col) in 4x4 matrix A given outer block ob and inner block ib."""
        r = (ob // 2) * 2 + (ib // 2)
        c = (ob % 2) * 2 + (ib % 2)
        return r, c

    def b_pos(ob, ib):
        """Position (row, col) in 4x5 matrix B given outer block ob and inner block ib."""
        r = (ob // 2) * 2 + (ib // 2)
        c = (ob % 2) * 2 + (ib % 2)
        return r, c

    def c_pos(ob, ib):
        """Position (row, col) in 4x4 matrix C given outer block ob and inner block ib."""
        r = (ob // 2) * 2 + (ib // 2)
        c = (ob % 2) * 2 + (ib % 2)
        return r, c

    for om in range(7):
        for im in range(7):
            col = om * 7 + im

            # Fill U (A-side)
            for (ab, ac) in SA[om]:
                for (ab2, ac2) in SA[im]:
                    r, c = a_pos(ab, ab2)
                    U[r * m + c, col] += ac * ac2

            # Fill V (B-side)
            for (bb, bc) in SB[om]:
                for (bb2, bc2) in SB[im]:
                    r, c = b_pos(bb, bb2)
                    V[r * p + c, col] += bc * bc2

            # Fill W (C-side)
            for (cb, cc) in SC[om].items():
                for (cb2, cc2) in SC[im].items():
                    r, c = c_pos(cb, cb2)
                    # W index: k*n + i where C[i,k], so i=r, k=c
                    W[c * n + r, col] += cc * cc2

    # Quick sanity check
    if time.time() - t0 < deadline - 10:
        A_t = np.random.randn(4, 4)
        B_t = np.random.randn(4, 5)
        C_expected = A_t @ B_t
        C_actual = np.zeros((4, 5))
        for r in range(R):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            # C[i,k] += sum_j u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
            # This is a tensor contraction
            for i in range(4):
                for k in range(5):
                    s = 0.0
                    for j in range(4):
                        s += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
                    C_actual[i, k] += s
        if not np.allclose(C_actual, C_expected, atol=1e-9):
            # Fallback: return schoolbook rank 80
            U_fb = np.zeros((16, 80), dtype=float)
            V_fb = np.zeros((20, 80), dtype=float)
            W_fb = np.zeros((20, 80), dtype=float)
            for idx in range(80):
                i = idx // 20
                rem = idx % 20
                j = rem // 5
                k = rem % 5
                U_fb[i * 4 + j, idx] = 1.0
                V_fb[j * 5 + k, idx] = 1.0
                W_fb[k * 4 + i, idx] = 1.0
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
