import numpy as np
import time

_start = time.time()
_TIME_LIMIT = 118.0  # safety margin before 120 s limit


def _time_up():
    return (time.time() - _start) > _TIME_LIMIT


def construct():
    n, m, p = 2, 3, 9

    # Hybrid design:
    #   - intermediate columns j in {0,1} : 2x2x9 block, tiled into
    #       four 2x2x2 Strassen blocks (4*7 = 28) + one 2x2x1 direct (4)
    #   - intermediate column  j = 2     : 2x1x9 direct (2*9 = 18)
    #   total rank = 28 + 4 + 18 = 50
    rank = 50

    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 27 rows
    W = [[0] * rank for _ in range(p * n)]  # 18 rows

    col = 0

    # ---- Part 1: A[:, 0:2] @ B[0:2, :]  (2x2x9) ----
    for b in range(4):
        if _time_up():
            return {"U": U, "V": V, "W": W}

        k0 = 2 * b
        k1 = 2 * b + 1

        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1

        u_a, u_b, u_c, u_d = 0, 1, 3, 4  # A00, A01, A10, A11

        # M1 = (a+d)(e+h)
        U[u_a][col] += 1; U[u_d][col] += 1
        V[v_e][col] += 1; V[v_h][col] += 1
        # M2 = (c+d)e
        U[u_c][col + 1] += 1; U[u_d][col + 1] += 1
        V[v_e][col + 1] += 1
        # M3 = a(f-h)
        U[u_a][col + 2] += 1
        V[v_f][col + 2] += 1; V[v_h][col + 2] -= 1
        # M4 = d(g-e)
        U[u_d][col + 3] += 1
        V[v_g][col + 3] += 1; V[v_e][col + 3] -= 1
        # M5 = (a+b)h
        U[u_a][col + 4] += 1; U[u_b][col + 4] += 1
        V[v_h][col + 4] += 1
        # M6 = (c-a)(e+f)
        U[u_c][col + 5] += 1; U[u_a][col + 5] -= 1
        V[v_e][col + 5] += 1; V[v_f][col + 5] += 1
        # M7 = (b-d)(g+h)
        U[u_b][col + 6] += 1; U[u_d][col + 6] -= 1
        V[v_g][col + 6] += 1; V[v_h][col + 6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        W[w_00][col] += 1
        W[w_00][col + 3] += 1
        W[w_00][col + 4] -= 1
        W[w_00][col + 6] += 1

        W[w_01][col + 2] += 1
        W[w_01][col + 4] += 1

        W[w_10][col + 1] += 1
        W[w_10][col + 3] += 1

        W[w_11][col] += 1
        W[w_11][col + 1] -= 1
        W[w_11][col + 2] += 1
        W[w_11][col + 5] += 1

        col += 7

    # ---- 2x2x1 direct for column k=8 ----
    k8 = 8
    v_e8 = 0 * p + k8
    v_g8 = 1 * p + k8

    U[0][col] += 1
    V[v_e8][col] += 1
    W[k8 * n + 0][col] += 1

    U[1][col + 1] += 1
    V[v_g8][col + 1] += 1
    W[k8 * n + 0][col + 1] += 1

    U[3][col + 2] += 1
    V[v_e8][col + 2] += 1
    W[k8 * n + 1][col + 2] += 1

    U[4][col + 3] += 1
    V[v_g8][col + 3] += 1
    W[k8 * n + 1][col + 3] += 1

    col += 4

    # ---- Part 2: A[:, 2:3] @ B[2, :]  (2x1x9 direct) ----
    for k in range(9):
        if _time_up():
            return {"U": U, "V": V, "W": W}

        v_b = 2 * p + k

        U[2][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1

        U[5][col + 1] += 1
        V[v_b][col + 1] += 1
        W[k * n + 1][col + 1] += 1

        col += 2

    # ---- Self-validation (cheap, integer-exact) ----
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)

    # tensor T[i, j, k] = U[i,j] * V[j,k] * W[k,i]
    T = np.einsum('ir,js,ks->ijks', Ua, Va, Wa, optimize=True)
    T = T.sum(axis=3)

    expected = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i, j, k] = 1.0

    if not np.allclose(T, expected, atol=1e-9):
        # Fallback: pure direct multiplication (rank = n*m*p = 54)
        U2 = [[0] * (n * m * p) for _ in range(n * m)]
        V2 = [[0] * (n * m * p) for _ in range(m * p)]
        W2 = [[0] * (n * m * p) for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U2[i * m + j][r] = 1
                    V2[j * p + k][r] = 1
                    W2[k * n + i][r] = 1
        return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    n, m, p = 2, 3, 9
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B

    A_flat = A.flatten()
    B_flat = B.flatten()

    C_flat = np.zeros(n * p)
    U_arr = np.array(res["U"])
    V_arr = np.array(res["V"])
    W_arr = np.array(res["W"])

    for s in range(U_arr.shape[1]):
        val = np.dot(U_arr[:, s], A_flat) * np.dot(V_arr[:, s], B_flat)
        C_flat += val * W_arr[:, s]

    C_actual = C_flat.reshape((p, n)).T

    if np.allclose(C_actual, C_expected, atol=1e-6):
        print("Validation Passed, rank =", U_arr.shape[1])
    else:
        print("Mismatch detected")
