import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Schoolbook fallback
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Strassen-based decomposition for rank 55
    def make_strassen():
        rank = 55
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        # Iterate over 5 blocks of 2 columns
        for t in range(5):
            k0 = 2 * t
            k1 = 2 * t + 1

            # M1: (A00 + A11) * (B0,k0 + B1,k1)
            # U: A00(1), A11(1)
            # V: B0,k0(1), B1,k1(1)
            # W: C0,k0(1), C1,k1(1)
            U[0 * 3 + 0][col] += 1
            U[1 * 3 + 1][col] += 1
            V[0 * 10 + k0][col] += 1
            V[1 * 10 + k1][col] += 1
            W[k0 * 2 + 0][col] += 1
            W[k1 * 2 + 1][col] += 1
            col += 1

            # M2: (A10 + A11) * B0,k0
            # U: A10(1), A11(1)
            # V: B0,k0(1)
            # W: C1,k0(1), C1,k1(-1)
            U[1 * 3 + 0][col] += 1
            U[1 * 3 + 1][col] += 1
            V[0 * 10 + k0][col] += 1
            W[k0 * 2 + 1][col] += 1
            W[k1 * 2 + 1][col] -= 1
            col += 1

            # M3: A00 * (B0,k1 - B1,k1)
            # U: A00(1)
            # V: B0,k1(1), B1,k1(-1)
            # W: C0,k1(1), C1,k1(1)
            U[0 * 3 + 0][col] += 1
            V[0 * 10 + k1][col] += 1
            V[1 * 10 + k1][col] -= 1
            W[k1 * 2 + 0][col] += 1
            W[k1 * 2 + 1][col] += 1
            col += 1

            # M4: A11 * (B1,k0 - B0,k0)
            # U: A11(1)
            # V: B1,k0(1), B0,k0(-1)
            # W: C0,k0(1), C1,k0(1)
            U[1 * 3 + 1][col] += 1
            V[1 * 10 + k0][col] += 1
            V[0 * 10 + k0][col] -= 1
            W[k0 * 2 + 0][col] += 1
            W[k0 * 2 + 1][col] += 1
            col += 1

            # M5: (A00 + A01) * B1,k1
            # U: A00(1), A01(1)
            # V: B1,k1(1)
            # W: C0,k0(-1), C0,k1(1)
            U[0 * 3 + 0][col] += 1
            U[0 * 3 + 1][col] += 1
            V[1 * 10 + k1][col] += 1
            W[k0 * 2 + 0][col] -= 1
            W[k1 * 2 + 0][col] += 1
            col += 1

            # M6: (A10 - A00) * (B0,k0 + B0,k1)
            # U: A10(1), A00(-1)
            # V: B0,k0(1), B0,k1(1)
            # W: C1,k1(1)
            U[1 * 3 + 0][col] += 1
            U[0 * 3 + 0][col] -= 1
            V[0 * 10 + k0][col] += 1
            V[0 * 10 + k1][col] += 1
            W[k1 * 2 + 1][col] += 1
            col += 1

            # M7: (A01 - A11) * (B1,k0 + B1,k1)
            # U: A01(1), A11(-1)
            # V: B1,k0(1), B1,k1(1)
            # W: C0,k0(1)
            U[0 * 3 + 1][col] += 1
            U[1 * 3 + 1][col] -= 1
            V[1 * 10 + k0][col] += 1
            V[1 * 10 + k1][col] += 1
            W[k0 * 2 + 0][col] += 1
            col += 1

            # W00: A02 * B2,k0
            # U: A02(1)
            # V: B2,k0(1)
            # W: C0,k0(1)
            U[0 * 3 + 2][col] += 1
            V[2 * 10 + k0][col] += 1
            W[k0 * 2 + 0][col] += 1
            col += 1

            # W01: A02 * B2,k1
            # U: A02(1)
            # V: B2,k1(1)
            # W: C0,k1(1)
            U[0 * 3 + 2][col] += 1
            V[2 * 10 + k1][col] += 1
            W[k1 * 2 + 0][col] += 1
            col += 1

            # W10: A12 * B2,k0
            # U: A12(1)
            # V: B2,k0(1)
            # W: C1,k0(1)
            U[1 * 3 + 2][col] += 1
            V[2 * 10 + k0][col] += 1
            W[k0 * 2 + 1][col] += 1
            col += 1

            # W11: A12 * B2,k1
            # U: A12(1)
            # V: B2,k1(1)
            # W: C1,k1(1)
            U[1 * 3 + 2][col] += 1
            V[2 * 10 + k1][col] += 1
            W[k1 * 2 + 1][col] += 1
            col += 1

        assert col == rank, f"Expected rank {rank}, got {col}"

        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if not np.isclose(lhs, 1.0, atol=1e-9):
                        return False
        return True

    try:
        result = make_strassen()
        if validate(result['U'], result['V'], result['W']):
            return result
    except Exception:
        pass

    return make_schoolbook()
