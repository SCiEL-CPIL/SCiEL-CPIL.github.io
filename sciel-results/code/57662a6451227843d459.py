"""Matrix multiplication 5x5x5 - Block decomposition with Strassen for 2x2x2 blocks."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _strassen_2x2():
    """
    Return the 7-term Strassen decomposition for 2x2 matrix multiplication.
    A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = A * B
    Local indices: A: a00=0, a01=1, a10=2, a11=3
                    B: b00=0, b01=1, b10=2, b11=3
                    C: c00=0, c01=1, c10=2, c11=3
    """
    terms = []
    # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
    terms.append(([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]))
    # M2 = (a10+a11)b00 -> C10 += M2, C11 -= M2
    terms.append(([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]))
    # M3 = a00(b01-b11) -> C01 += M3, C11 += M3
    terms.append(([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]))
    # M4 = a11(b10-b00) -> C00 += M4, C10 += M4
    terms.append(([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]))
    # M5 = (a00+a01)b11 -> C00 -= M5, C01 += M5
    terms.append(([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]))
    # M6 = (a10-a00)(b00+b01) -> C11 += M6
    terms.append(([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
    # M7 = (a01-a11)(b10+b11) -> C00 += M7
    terms.append(([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]))
    return terms


def _verify_decomposition(U, V, W, n=5, m=5, p=5, tol=1e-10):
    """Verify that the tensor decomposition is exact using random test matrices."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B

    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            u_vals = np.zeros(rank)
            for j in range(m):
                u_vals += U[i * m + j, :] * A[i, j]
            v_vals = np.zeros(rank)
            for j in range(m):
                v_vals += V[j * p + k, :] * B[j, k]
            C_decomp[i, k] = np.dot(W[k * n + i, :], u_vals * v_vals)

    err = np.max(np.abs(C_decomp - C_expected))
    return err < tol, err


def _build_decomposition(n=5, m=5, p=5):
    """
    Build block decomposition of 5x5x5 matrix multiplication.
    Split 5 into 2+2+1. Use Strassen (rank 7) for 2x2 blocks, schoolbook (rank s^3) for others.
    """
    splits = [0, 2, 4, 5]
    nb = len(splits) - 1  # number of blocks = 3

    # Precompute block dimensions
    A_dims = {}  # (i, j) -> (rows, cols)
    for i in range(nb):
        for j in range(nb):
            A_dims[(i, j)] = (splits[i+1] - splits[i], splits[j+1] - splits[j])

    B_dims = {}  # (j, k) -> (rows, cols)
    for j in range(nb):
        for k in range(nb):
            B_dims[(j, k)] = (splits[j+1] - splits[j], splits[k+1] - splits[k])

    C_dims = {}  # (i, k) -> (rows, cols)
    for i in range(nb):
        for k in range(nb):
            C_dims[(i, k)] = (splits[i+1] - splits[i], splits[k+1] - splits[k])

    strassen = _strassen_2x2()
    terms = []  # list of (u_dict, v_dict, w_dict)

    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                ha, wa = A_dims[(i, j)]
                hb, wb = B_dims[(j, k)]
                hc, wc = C_dims[(i, k)]

                assert ha == hc, f"A rows {ha} != C rows {hc} at blocks ({i},{j},{k})"
                assert wa == hb, f"A cols {wa} != B rows {hb} at blocks ({i},{j},{k})"

                a_r0 = splits[i]
                a_c0 = splits[j]
                b_r0 = splits[j]
                b_c0 = splits[k]
                c_r0 = splits[i]
                c_c0 = splits[k]

                if ha == 2 and wa == 2 and hb == 2 and wb == 2:
                    # Use Strassen for 2x2x2 block
                    for u_loc, u_c, v_loc, v_c, w_loc, w_c in strassen:
                        u_dict = {}
                        for li, coeff in zip(u_loc, u_c):
                            if coeff != 0:
                                li_r, li_c = li // 2, li % 2
                                gu = (a_r0 + li_r) * m + (a_c0 + li_c)
                                u_dict[gu] = u_dict.get(gu, 0) + coeff

                        v_dict = {}
                        for li, coeff in zip(v_loc, v_c):
                            if coeff != 0:
                                li_r, li_c = li // 2, li % 2
                                gv = (b_r0 + li_r) * p + (b_c0 + li_c)
                                v_dict[gv] = v_dict.get(gv, 0) + coeff

                        w_dict = {}
                        for li, coeff in zip(w_loc, w_c):
                            if coeff != 0:
                                li_r, li_c = li // 2, li % 2
                                gw = (c_c0 + li_c) * n + (c_r0 + li_r)
                                w_dict[gw] = w_dict.get(gw, 0) + coeff

                        terms.append((u_dict, v_dict, w_dict))
                else:
                    # Schoolbook for non-2x2 blocks
                    for li in range(ha):
                        for lj in range(wa):
                            for lk in range(wb):
                                gu = (a_r0 + li) * m + (a_c0 + lj)
                                gv = (b_r0 + lj) * p + (b_c0 + lk)
                                gw = (c_c0 + lk) * n + (c_r0 + li)
                                terms.append(({gu: 1}, {gv: 1}, {gw: 1}))

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            if 0 <= idx < n * m:
                U[idx][c] = coeff
        for idx, coeff in v_dict.items():
            if 0 <= idx < m * p:
                V[idx][c] = coeff
        for idx, coeff in w_dict.items():
            if 0 <= idx < p * n:
                W[idx][c] = coeff

    return U, V, W


def _schoolbook(n=5, m=5, p=5):
    """Fallback schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Build block decomposition
    U, V, W = _build_decomposition(n, m, p)
    rank = len(U[0]) if U else 0

    # Verify
    valid, err = _verify_decomposition(U, V, W, n, m, p)

    if not valid:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)

    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        # Very close to deadline, return immediately
        return {"U": U, "V": W, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
