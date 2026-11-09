import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100  # 100s safety margin before 150s kill

    n, m, p = DIMENSIONS

    terms = []

    def add_term(u_vals, v_vals, w_vals):
        terms.append((u_vals, v_vals, w_vals))

    def strassen_222(i_off, j_off, k_off):
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        add_term({(i_off, j_off): 1, (i_off+1, j_off+1): 1},
                 {(j_off, k_off): 1, (j_off+1, k_off+1): 1},
                 {(k_off, i_off): 1, (k_off+1, i_off+1): 1})
        # M2 = (A10+A11)B00 -> C10
        add_term({(i_off+1, j_off): 1, (i_off+1, j_off+1): 1},
                 {(j_off, k_off): 1},
                 {(k_off, i_off+1): 1})
        # M3 = A00(B10-B11) -> C01, C11
        add_term({(i_off, j_off): 1},
                 {(j_off+1, k_off): 1, (j_off+1, k_off+1): -1},
                 {(k_off+1, i_off): 1, (k_off+1, i_off+1): 1})
        # M4 = A11(B01-B00) -> C00, C10
        add_term({(i_off+1, j_off+1): 1},
                 {(j_off, k_off+1): 1, (j_off, k_off): -1},
                 {(k_off, i_off): 1, (k_off, i_off+1): 1})
        # M5 = (A00+A01)B11 -> C01
        add_term({(i_off, j_off): 1, (i_off, j_off+1): 1},
                 {(j_off+1, k_off+1): 1},
                 {(k_off+1, i_off): 1})
        # M6 = (A10-A00)(B00+B01) -> C10, C11
        add_term({(i_off+1, j_off): 1, (i_off, j_off): -1},
                 {(j_off, k_off): 1, (j_off, k_off+1): 1},
                 {(k_off, i_off+1): 1, (k_off+1, i_off+1): 1})
        # M7 = (A01-A11)(B10+B11) -> C00, C01
        add_term({(i_off, j_off+1): 1, (i_off+1, j_off+1): -1},
                 {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1},
                 {(k_off, i_off): 1, (k_off+1, i_off): 1})

    def schoolbook(i, j, k):
        add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    used = set()
    for i in range(0, n, 2):
        for j in range(0, m, 2):
            for k in range(0, p, 2):
                if i+1 < n and j+1 < m and k+1 < p:
                    strassen_222(i, j, k)
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                used.add((i+di, j+dj, k+dk))
                else:
                    for di in range(n - i):
                        for dj in range(m - j):
                            for dk in range(p - k):
                                if (i+di, j+dj, k+dk) not in used:
                                    schoolbook(i+di, j+dj, k+dk)
                                    used.add((i+di, j+dj, k+dk))

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r, (u_vals, v_vals, w_vals) in enumerate(terms):
        for (i, j), val in u_vals.items():
            U[i * m + j][r] = val
        for (j, k), val in v_vals.items():
            V[j * p + k][r] = val
        for (k, i), val in w_vals.items():
            W[k * n + i][r] = val

    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    val = float(np.dot(U_arr[i*m+j], np.multiply(V_arr[jp*p+k], W_arr[k*n+i])))
                    expected = 1.0 if j == jp else 0.0
                    err = abs(val - expected)
                    if err > max_err:
                        max_err = err

    if max_err > 1e-9:
        # Fallback to schoolbook
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
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
