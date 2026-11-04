import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    def make_rank_55():
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        idx = 0
        # Part 1: Strassen for 2x2 x 2x10 (5 blocks of 2x2x2)
        for block in range(5):
            k0 = 2 * block
            k1 = 2 * block + 1

            # M1 = (A00 + A11) * (B0,k0 + B1,k1)
            U[0 * 3 + 0, idx] = 1.0
            U[1 * 3 + 1, idx] = 1.0
            V[0 * 10 + k0, idx] = 1.0
            V[1 * 10 + k1, idx] = 1.0
            W[k0 * 2 + 0, idx] = 1.0
            W[k1 * 2 + 1, idx] = 1.0
            idx += 1

            # M2 = (A10 + A11) * B0,k0
            U[1 * 3 + 0, idx] = 1.0
            U[1 * 3 + 1, idx] = 1.0
            V[0 * 10 + k0, idx] = 1.0
            W[k0 * 2 + 1, idx] = 1.0
            W[k1 * 2 + 1, idx] = -1.0
            idx += 1

            # M3 = A00 * (B0,k1 - B1,k1)
            U[0 * 3 + 0, idx] = 1.0
            V[0 * 10 + k1, idx] = 1.0
            V[1 * 10 + k1, idx] = -1.0
            W[k1 * 2 + 0, idx] = 1.0
            W[k1 * 2 + 1, idx] = 1.0
            idx += 1

            # M4 = A11 * (B1,k0 - B0,k0)
            U[1 * 3 + 1, idx] = 1.0
            V[1 * 10 + k0, idx] = 1.0
            V[0 * 10 + k0, idx] = -1.0
            W[k0 * 2 + 0, idx] = 1.0
            W[k0 * 2 + 1, idx] = 1.0
            idx += 1

            # M5 = (A00 + A01) * B1,k1
            U[0 * 3 + 0, idx] = 1.0
            U[0 * 3 + 1, idx] = 1.0
            V[1 * 10 + k1, idx] = 1.0
            W[k0 * 2 + 0, idx] = -1.0
            W[k1 * 2 + 0, idx] = 1.0
            idx += 1

            # M6 = (A10 - A00) * (B0,k0 + B0,k1)
            U[1 * 3 + 0, idx] = 1.0
            U[0 * 3 + 0, idx] = -1.0
            V[0 * 10 + k0, idx] = 1.0
            V[0 * 10 + k1, idx] = 1.0
            W[k1 * 2 + 1, idx] = 1.0
            idx += 1

            # M7 = (A01 - A11) * (B1,k0 + B1,k1)
            U[0 * 3 + 1, idx] = 1.0
            U[1 * 3 + 1, idx] = -1.0
            V[1 * 10 + k0, idx] = 1.0
            V[1 * 10 + k1, idx] = 1.0
            W[k0 * 2 + 0, idx] = 1.0
            idx += 1

        # Part 2: Schoolbook for 2x1 x 1x10
        for i in range(2):
            for k in range(10):
                U[i * 3 + 2, idx] = 1.0
                V[2 * 10 + k, idx] = 1.0
                W[k * 2 + i, idx] = 1.0
                idx += 1

        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Check a few random points for speed, or all if fast enough
        # 2*3*10 = 60 checks, dot product of length 55. Very fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if not np.isclose(lhs, 1.0, atol=1e-9):
                        return False
        return True

    try:
        result = make_rank_55()
        if validate(result['U'], result['V'], result['W']):
            return {"U": result['U'].tolist(), "V": result['V'].tolist(), "W": result['W'].tolist()}
    except Exception:
        pass

    # Fallback: Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
