import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Build the rank-50 decomposition deterministically via block-structure
    # 6x Strassen(2,2,2) + 2x Schoolbook(2,2,1)
    rank_target = 50

    U = [[0] * rank_target for _ in range(n * m)]
    V = [[0] * rank_target for _ in range(m * p)]
    W = [[0] * rank_target for _ in range(p * n)]

    col = 0

    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= rank_target:
            return False

        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)

        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)

        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1

        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1

        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1

        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1

        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1

        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[c11][col] += 1
        col += 1

        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        col += 1

        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= rank_target:
            return False

        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)

        e = j0 * p + k0
        g = (j0 + 1) * p + k0

        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)

        U[a][col] += 1
        V[e][col] += 1
        W[c00][col] += 1
        col += 1

        U[b][col] += 1
        V[g][col] += 1
        W[c00][col] += 1
        col += 1

        U[c][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        col += 1

        U[d][col] += 1
        V[g][col] += 1
        W[c10][col] += 1
        col += 1

        return True

    # Assemble: 3 Strassen blocks in (0,0,k) for k=0,2,4; schoolbook for k=6
    for k0 in [0, 2, 4]:
        if time.time() > deadline:
            break
        if not add_strassen_222(0, 0, k0):
            break

    if time.time() <= deadline:
        if not add_schoolbook_221(0, 0, 6):
            pass

    # Second row-block of A (rows 0-1, cols 2-3 of A x rows 2-3 of B)
    for k0 in [0, 2, 4]:
        if time.time() > deadline:
            break
        if not add_strassen_222(0, 2, k0):
            break

    if time.time() <= deadline:
        if not add_schoolbook_221(0, 2, 6):
            pass

    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                rec = float(np.dot(U_arr[a_idx, :], V_arr[b_idx, :]))
                rec2 = 0.0
                for r in range(rank_target):
                    rec2 += U_arr[a_idx, r] * V_arr[b_idx, r] * W_arr[c_idx, r]
                if abs(rec2 - 1.0) > max_err:
                    max_err = abs(rec2 - 1.0)

    if max_err > 1e-9:
        # Fallback: return schoolbook decomposition (rank = n*m*p = 56)
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_fb[a_idx][r] = 1
                    V_fb[b_idx][r] = 1
                    W_fb[c_idx][r] = 1
                    r += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
