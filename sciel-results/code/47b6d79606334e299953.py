"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 2, 13

    # Schoolbook fallback: rank 52
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_school[i * m + k, col] = 1.0
                V_school[k * p + j, col] = 1.0
                W_school[j * n + i, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    try:
        # Strassen-based: 6 blocks of 2x2 (rank 7 each) + 1 block of 2x1 (rank 4)
        # Total: 6*7 + 4 = 46
        
        rank_total = 6 * 7 + 4
        U = np.zeros((n * m, rank_total))
        V = np.zeros((m * p, rank_total))
        W = np.zeros((p * n, rank_total))

        # Strassen terms for 2x2x2x2
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = A*B
        # Strassen's 7 products:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b10)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # U vectors (for A coefficients [a00, a01, a10, a11]):
        # M1: a00 + a11 -> [1, 0, 0, 1]
        # M2: a10 + a11 -> [0, 0, 1, 1]
        # M3: a00 -> [1, 0, 0, 0]
        # M4: a11 -> [0, 0, 0, 1]
        # M5: a00 + a01 -> [1, 1, 0, 0]
        # M6: a10 - a00 -> [-1, 0, 1, 0]
        # M7: a01 - a11 -> [0, 1, 0, -1]

        # V vectors (for B coefficients [b00, b01, b10, b11]):
        # M1: b00 + b11 -> [1, 0, 0, 1]
        # M2: b00 -> [1, 0, 0, 0]
        # M3: b01 - b10 -> [0, 1, -1, 0]
        # M4: b10 - b00 -> [-1, 0, 1, 0]
        # M5: b11 -> [0, 0, 0, 1]
        # M6: b00 + b01 -> [1, 1, 0, 0]
        # M7: b10 + b11 -> [0, 0, 1, 1]

        # W vectors (for C coefficients [c00, c01, c10, c11]):
        # M1: c00 + c11 -> [1, 0, 0, 1]
        # M2: -c10 + c11 -> [0, 0, -1, 1]
        # M3: c01 + c10 -> [0, 1, 1, 0]
        # M4: c00 + c10 -> [1, 0, 1, 0]
        # M5: -c00 + c01 -> [-1, 1, 0, 0]
        # M6: c11 -> [0, 0, 0, 1]
        # M7: c00 -> [1, 0, 0, 0]

        strassen_terms = [
            (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
            (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, -1, 1])),
            (np.array([1, 0, 0, 0]), np.array([0, 1, -1, 0]), np.array([0, 1, 1, 0])),
            (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 0, 1, 0])),
            (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
            (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
            (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),
        ]

        col = 0
        # 6 blocks of 2 columns each (j=0..11)
        for block in range(6):
            j0 = block * 2
            j1 = block * 2 + 1

            for t in range(7):
                u_local, v_local, w_local = strassen_terms[t]

                U[0, col] = u_local[0]
                U[1, col] = u_local[1]
                U[2, col] = u_local[2]
                U[3, col] = u_local[3]

                V[0 * p + j0, col] = v_local[0]
                V[0 * p + j1, col] = v_local[1]
                V[1 * p + j0, col] = v_local[2]
                V[1 * p + j1, col] = v_local[3]

                W[j0 * n + 0, col] = w_local[0]
                W[j0 * n + 1, col] = w_local[1]
                W[j1 * n + 0, col] = w_local[2]
                W[j1 * n + 1, col] = w_local[3]

                col += 1

        # Last column (j=12): schoolbook 2x2 x 2x1, rank 4
        j_last = 12
        for i in range(n):
            for k in range(m):
                U[i * m + k, col] = 1.0
                V[k * p + j_last, col] = 1.0
                W[j_last * n + i, col] = 1.0
                col += 1

        # Verify correctness
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_target[i * m + k, k * p + j, j * n + i] = 1.0

        T_approx = np.einsum('it,jt,kt->ijk', U, V, W)
        error = np.max(np.abs(T_approx - T_target))

        if error < 1e-10:
            best_rank = rank_total
            best_U, best_V, best_W = U, V, W

    except Exception:
        pass

    if time.time() > deadline:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
