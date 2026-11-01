import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=int)
    V = np.zeros((m * p, n * m * p), dtype=int)
    W = np.zeros((p * n, n * m * p), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * (m * p) + j * p + k
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
    return U, V, W


def _strassen_terms():
    return [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]


def _build_decomp(n, m, p):
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    strassen = _strassen_terms()

    # 4 blocks: (A_col_block, B_row_block) for the 2x2 subproduct
    # C_col_block = 0 for left half of C, 1 for right half
    # block 0: A[:, 0:2] @ B[0:2, 0:2] -> C[:, 0:2]
    # block 1: A[:, 2:4] @ B[2:4, 0:2] -> C[:, 0:2]
    # block 2: A[:, 0:2] @ B[0:2, 2:4] -> C[:, 2:4]
    # block 3: A[:, 2:4] @ B[2:4, 2:4] -> C[:, 2:4]

    for block in range(4):
        A_cb = block % 2
        B_rb = block // 2
        C_cb = 0 if block < 2 else 1

        for r in range(7):
            col = block * 7 + r
            u_sub, v_sub, w_sub = strassen[r]

            for a_idx, cu in u_sub.items():
                ar = a_idx // 2
                ac = a_idx % 2
                g_col = 2 * A_cb + ac
                U[ar * m + g_col, col] += cu

            for b_idx, cv in v_sub.items():
                br = b_idx // 2
                bc = b_idx % 2
                g_row = 2 * B_rb + br
                g_col = 2 * C_cb + bc
                V[g_row * p + g_col, col] += cv

            for c_idx, cw in w_sub.items():
                cr = c_idx // 2
                cc = c_idx % 2
                W[(2 * C_cb + cc) * n + cr, col] += cw

    return U, V, W


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    A_test = np.arange(n * m).reshape(n, m, dtype=float) + 1.0
    B_test = np.arange(m * p).reshape(m, p, dtype=float) + 2.0
    C_test = A_test @ B_test
    rank = U.shape[1]
    Cf = np.zeros((n, p), dtype=float)
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(rank):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    return np.allclose(C_test, Cf, atol=1e-9)


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()

    U, V, W = _build_decomp(n, m, p)

    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U_s, V_s, W_s = _schoolbook(n, m, p)
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
