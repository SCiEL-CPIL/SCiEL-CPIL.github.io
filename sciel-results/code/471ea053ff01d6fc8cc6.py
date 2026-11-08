import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 140.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros(n * m, dtype=float)
        V = np.zeros(m * p, dtype=float)
        W = np.zeros(p * n, dtype=float)
        for (i, j), val in u_vals.items():
            U[i * m + j] = val
        for (j, k), val in v_vals.items():
            V[j * p + k] = val
        for (k, i), val in w_vals.items():
            W[k * n + i] = val
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        add_term({(i0, j0): 1, (i1, j1): 1}, {(j0, k0): 1, (j1, k1): 1}, {(k0, i0): 1, (k1, i1): 1})
        add_term({(i1, j0): 1, (i1, j1): 1}, {(j0, k0): 1}, {(k0, i1): 1, (k1, i1): -1})
        add_term({(i0, j0): 1}, {(j0, k1): 1, (j1, k1): -1}, {(k1, i0): 1, (k1, i1): 1})
        add_term({(i1, j1): 1}, {(j1, k0): 1, (j0, k0): -1}, {(k0, i0): 1, (k0, i1): 1})
        add_term({(i0, j0): 1, (i0, j1): 1}, {(j1, k1): 1}, {(k0, i0): -1, (k1, i0): 1})
        add_term({(i1, j0): 1, (i0, j0): -1}, {(j0, k0): 1, (j0, k1): 1}, {(k1, i1): 1})
        add_term({(i0, j1): 1, (i1, j1): -1}, {(j1, k0): 1, (j1, k1): 1}, {(k0, i0): 1})

    def mm_2x1_1x2(i0, i1, j, k0, k1):
        for i in [i0, i1]:
            for k in [k0, k1]:
                add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    def mm_2x2_2x3(i0, i1, j0, j1, k0, k1, k2):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for i in [i0, i1]:
            for j in [j0, j1]:
                add_term({(i, j): 1}, {(j, k2): 1}, {(k2, i): 1})

    def mm_2x3_3x2(i0, i1, j0, j1, j2, k0, k1):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        mm_2x1_1x2(i0, i1, j2, k0, k1)

    def mm_3x2_2x2(i0, i1, i2, j0, j1, k0, k1):
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        for j in [j0, j1]:
            for k in [k0, k1]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})

    def mm_3x2_2x3(i0, i1, i2, j0, j1, k0, k1, k2):
        mm_2x2_2x3(i0, i1, j0, j1, k0, k1, k2)
        for j in [j0, j1]:
            for k in [k0, k1, k2]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})

    def mm_3x3_3x2(i0, i1, i2, j0, j1, j2, k0, k1):
        mm_3x2_2x2(i0, i1, i2, j0, j1, k0, k1)
        for i in [i0, i1, i2]:
            for k in [k0, k1]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})

    def mm_3x3_3x3(i0, i1, i2, j0, j1, j2, k0, k1, k2):
        mm_3x2_2x3(i0, i1, i2, j0, j1, k0, k1, k2)
        for i in [i0, i1, i2]:
            for k in [k0, k1, k2]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})

    # 2+3 block decomposition
    strassen_2x2(0, 1, 0, 1, 0, 1)
    mm_2x3_3x2(0, 1, 2, 3, 4, 0, 1)
    mm_2x2_2x3(0, 1, 0, 1, 2, 3, 4)
    mm_2x3_3x3(0, 1, 2, 3, 4, 2, 3, 4)
    mm_3x2_2x2(2, 3, 4, 0, 1, 0, 1)
    mm_3x3_3x2(2, 3, 4, 2, 3, 4, 0, 1)
    mm_3x2_2x3(2, 3, 4, 0, 1, 2, 3, 4)
    mm_3x3_3x3(2, 3, 4, 2, 3, 4, 2, 3, 4)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    rank = U.shape[1]

    if time.time() - t0 < deadline - 5:
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], np.multiply(V[j*p+k, :], W[k*n+i, :]))
                    if abs(val - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            U = np.zeros((n*m, n*m*p))
            V = np.zeros((m*p, n*m*p))
            W = np.zeros((p*n, n*m*p))
            for i2 in range(n):
                for j2 in range(m):
                    for k2 in range(p):
                        idx = i2*m*p + j2*p + k2
                        U[i2*m+j2, idx] = 1
                        V[j2*p+k2, idx] = 1
                        W[k2*n+i2, idx] = 1
            rank = n*m*p

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
