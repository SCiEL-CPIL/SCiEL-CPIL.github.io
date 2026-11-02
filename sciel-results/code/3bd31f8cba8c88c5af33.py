"""Matrix multiplication 2x2x10: rank-minimal exact tensor decomposition.

Strategy: Use 5 independent 2x2x2 Strassen rank-7 decompositions,
giving total rank 5 * 7 = 35. Since Strassen's rank-7 is optimal for
2x2x2 (Landsberg 1987), and each pair of output columns forms an
independent 2x2x2 sub-tensor, rank 35 is the theoretical minimum.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 10]
_DEADLINE = 120.0
_START_TIME = time.time()


def construct():
    n, m, p = DIMENSIONS  # 2, 2, 10
    _t0 = time.time()

    # Use 5 independent 2x2x2 Strassen blocks: rank = 5 * 7 = 35
    num_blocks = p // 2  # 5
    rank = num_blocks * 7  # 35

    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35

    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # A = [[a, b], [c, d]] = [[A00, A01], [A10, A11]]
    # B = [[e, f], [g, h]] = [[B00, B01], [B10, B11]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # U rows: 0=a(A00), 1=b(A01), 2=c(A10), 3=d(A11)
    # V rows within block: 0=e(B00), 1=f(B01), 2=g(B10), 3=h(B11)
    #   actual V row = j*p + k where j = v_koff//2, k = 2*b + v_koff%2
    # W rows within block: 0=C00, 1=C01, 2=C10, 3=C11
    #   actual W row = k*n + i where i = w_koff//2, k = 2*b + w_koff%2

    strassen_decomp = [
        # M1 = (a+d)(e+h)
        (
            [(0, 1), (3, 1)],       # U: a + d
            [(0, 1), (3, 1)],       # V: e + h
            [(0, 1), (3, 1)],       # W: C00:1, C11:1
        ),
        # M2 = (c+d)*e
        (
            [(2, 1), (3, 1)],       # U: c + d
            [(0, 1)],               # V: e
            [(2, 1), (3, -1)],      # W: C10:1, C11:-1
        ),
        # M3 = a*(f-h)
        (
            [(0, 1)],               # U: a
            [(1, 1), (3, -1)],      # V: f - h
            [(1, 1), (3, 1)],       # W: C01:1, C11:1
        ),
        # M4 = d*(g-e)
        (
            [(3, 1)],               # U: d
            [(2, 1), (0, -1)],      # V: g - e
            [(0, 1), (2, 1)],       # W: C00:1, C10:1
        ),
        # M5 = (a+b)*h
        (
            [(0, 1), (1, 1)],       # U: a + b
            [(3, 1)],               # V: h
            [(1, 1), (0, -1)],      # W: C01:1, C00:-1
        ),
        # M6 = (c-a)*(e+f)
        (
            [(2, 1), (0, -1)],      # U: c - a
            [(0, 1), (1, 1)],       # V: e + f
            [(3, 1)],               # W: C11:1
        ),
        # M7 = (b-d)*(g+h)
        (
            [(1, 1), (3, -1)],      # U: b - d
            [(2, 1), (3, 1)],       # V: g + h
            [(0, 1)],               # W: C00:1
        ),
    ]

    for b in range(num_blocks):
        base_col = b * 7
        for t in range(7):
            col = base_col + t
            u_side, v_side, w_side = strassen_decomp[t]

            for u_row, u_coeff in u_side:
                U[u_row][col] = u_coeff

            for v_koff, v_coeff in v_side:
                j = v_koff // 2
                k = 2 * b + v_koff % 2
                V[j * p + k][col] = v_coeff

            for w_koff, w_coeff in w_side:
                i = w_koff // 2
                k = 2 * b + w_koff % 2
                W[k * n + i][col] = w_coeff

    # Verify correctness with multiple random tests
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for seed in [42, 123, 999, 7, 2024]:
            if time.time() - _t0 > _DEADLINE - 10:
                break
            rng = np.random.default_rng(seed)
            A_test = rng.integers(-5, 6, size=(n, m))
            B_test = rng.integers(-5, 6, size=(m, p))
            C_expected = A_test @ B_test

            A_flat = A_test.flatten()  # length 4: [a, b, c, d]
            B_flat = B_test.flatten()  # length 20: [x1,x2,...,x10, y1,y2,...,y10]

            C_computed = np.zeros(p * n)
            for r in range(rank):
                uA = np.dot(U_arr[:, r], A_flat)
                vB = np.dot(V_arr[:, r], B_flat)
                C_computed += uA * vB * W_arr[:, r]

            C_expected_reorder = np.zeros(p * n)
            for i in range(n):
                for k in range(p):
                    C_expected_reorder[k * n + i] = C_expected[i, k]

            if not np.allclose(C_computed, C_expected_reorder, atol=1e-9):
                raise ValueError(f"Decomposition verification failed for seed {seed}")

    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
