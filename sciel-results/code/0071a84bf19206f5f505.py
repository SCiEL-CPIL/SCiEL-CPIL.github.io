import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 2, 16
    num_blocks = p // m  # 8

    # Rank-7 Strassen decomposition for 2x2x2
    U_s = np.zeros((4, 7), dtype=np.int64)
    V_s = np.zeros((4, 7), dtype=np.int64)
    W_s = np.zeros((4, 7), dtype=np.int64)

    def add_u(i, j, coeff, col):
        U_s[i * 2 + j, col] += coeff

    def add_v(j, k, coeff, col):
        V_s[j * 2 + k, col] += coeff

    def add_w(k, i, coeff, col):
        W_s[k * 2 + i, col] += coeff

    # m1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    add_u(0, 0, 1, 0); add_u(1, 1, 1, 0)
    add_v(0, 0, 1, 0); add_v(1, 1, 1, 0)
    add_w(0, 0, 1, 0); add_w(1, 1, 1, 0)

    # m2 = (A10+A11)B00 -> C10 += 1, C11 += -1
    add_u(1, 0, 1, 1); add_u(1, 1, 1, 1)
    add_v(0, 0, 1, 1)
    add_w(0, 1, 1, 1); add_w(1, 1, -1, 1)

    # m3 = A00(B01-B11) -> C01 += 1, C11 += 1
    add_u(0, 0, 1, 2)
    add_v(0, 1, 1, 2); add_v(1, 1, -1, 2)
    add_w(1, 0, 1, 2); add_w(1, 1, 1, 2)

    # m4 = A11(B10-B00) -> C00 += 1, C10 += 1
    add_u(1, 1, 1, 3)
    add_v(1, 0, 1, 3); add_v(0, 0, -1, 3)
    add_w(0, 0, 1, 3); add_w(0, 1, 1, 3)

    # m5 = (A00+A01)B11 -> C00 += -1, C01 += 1
    add_u(0, 0, 1, 4); add_u(0, 1, 1, 4)
    add_v(1, 1, 1, 4)
    add_w(0, 0, -1, 4); add_w(1, 0, 1, 4)

    # m6 = (A10-A00)(B00+B01) -> C11 += 1
    add_u(1, 0, 1, 5); add_u(0, 0, -1, 5)
    add_v(0, 0, 1, 5); add_v(0, 1, 1, 5)
    add_w(1, 1, 1, 5)

    # m7 = (A01-A11)(B10+B11) -> C00 += 1
    add_u(0, 1, 1, 6); add_u(1, 1, -1, 6)
    add_v(1, 0, 1, 6); add_v(1, 1, 1, 6)
    add_w(0, 0, 1, 6)

    # Expand to 2x2x16
    # U: 4x7 -> 4x56 (repeat across 8 blocks)
    U = np.tile(U_s, (1, num_blocks))

    # V: 4x7 -> 32x56
    V = np.zeros((m * p, 7 * num_blocks), dtype=np.int64)
    for t in range(num_blocks):
        cols = slice(7 * t, 7 * t + 7)
        V[2 * t, cols] = V_s[0, :]
        V[2 * t + 1, cols] = V_s[1, :]
        V[16 + 2 * t, cols] = V_s[2, :]
        V[17 + 2 * t, cols] = V_s[3, :]

    # W: 4x7 -> 32x56
    W = np.zeros((p * n, 7 * num_blocks), dtype=np.int64)
    for t in range(num_blocks):
        cols = slice(7 * t, 7 * t + 7)
        W[4 * t, cols] = W_s[0, :]
        W[4 * t + 1, cols] = W_s[1, :]
        W[4 * t + 2, cols] = W_s[2, :]
        W[4 * t + 3, cols] = W_s[3, :]

    # Verify correctness with a random test
    if time.time() < deadline - 2.0:
        try:
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test

            A_flat = A_test.flatten(order='F')  # i*m + j
            B_flat = B_test.flatten(order='F')  # j*p + k

            U_A = U @ A_flat
            V_B = V @ B_flat
            C_flat = W.T @ (U_A * V_B)

            # C_flat index = k*n + i, so reshape to (p, n) with order='F' then transpose
            C_reconstructed = C_flat.reshape(p, n, order='F').T
            max_err = np.max(np.abs(C_reconstructed - C_expected))
            if max_err > 1e-10:
                # Fallback: use schoolbook
                U = np.zeros((n * m, n * m * p), dtype=np.int64)
                V = np.zeros((m * p, n * m * p), dtype=np.int64)
                W = np.zeros((p * n, n * m * p), dtype=np.int64)
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j, col] = 1
                            V[j * p + k, col] = 1
                            W[k * n + i, col] = 1
                            col += 1
        except Exception:
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
