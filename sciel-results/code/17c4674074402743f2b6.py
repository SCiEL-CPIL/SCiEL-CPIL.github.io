import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    n, m, p = 3, 4, 6

    def time_ok():
        return time.time() - start_time < DEADLINE

    terms = []

    # Row 0: schoolbook for 1x4x6 = 24 terms
    for j in range(m):
        for k in range(p):
            terms.append(
                (
                    {(0, j): 1},
                    {(j, k): 1},
                    {(k, 0): 1}
                )
            )

    # Rows 1-2: Strassen for 2x2x2 blocks
    # m=4 split into [0,2] and [2,4]
    # p=6 split into [0,2], [2,4], [4,6]
    for j1 in [0, 2]:
        for k1 in [0, 2, 4]:
            i1, i2 = 1, 2
            j2 = j1 + 1
            k2 = k1 + 1

            # M1 = (a+d)(e+h)
            terms.append(
                (
                    {(i1, j1): 1, (i2, j1): 1},
                    {(j1, k1): 1, (j2, k1): 1},
                    {(k1, i1): 1, (k1, i2): 1}
                )
            )
            # M2 = (c+d)e
            terms.append(
                (
                    {(i2, j1): 1, (i2, j2): 1},
                    {(j1, k1): 1},
                    {(k1, i2): 1, (k2, i2): -1}
                )
            )
            # M3 = a(f-h)
            terms.append(
                (
                    {(i1, j1): 1},
                    {(j1, k2): 1, (j2, k2): -1},
                    {(k2, i1): 1, (k2, i2): 1}
                )
            )
            # M4 = d(g-e)
            terms.append(
                (
                    {(i2, j2): 1},
                    {(j2, k1): 1, (j1, k1): -1},
                    {(k1, i1): 1, (k1, i2): 1}
                )
            )
            # M5 = (a+b)h
            terms.append(
                (
                    {(i1, j1): 1, (i1, j2): 1},
                    {(j2, k1): 1},
                    {(k1, i1): -1, (k2, i1): 1}
                )
            )
            # M6 = (c-a)(e+f)
            terms.append(
                (
                    {(i2, j1): 1, (i1, j1): -1},
                    {(j1, k1): 1, (j1, k2): 1},
                    {(k2, i2): 1}
                )
            )
            # M7 = (b-d)(g+h)
            terms.append(
                (
                    {(i1, j2): 1, (i2, j2): -1},
                    {(j2, k1): 1, (j2, k2): 1},
                    {(k1, i1): 1}
                )
            )

    rank = len(terms)

    def terms_to_matrices(terms):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = coeff

        return U, V, W

    def verify_decomposition(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        rank = U_arr.shape[1]

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    U, V, W = terms_to_matrices(terms)

    if not verify_decomposition(U, V, W):
        # Fallback: pure schoolbook
        terms_orig = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms_orig.append(
                        (
                            {(i, j): 1},
                            {(j, k): 1},
                            {(k, i): 1}
                        )
                    )
        U, V, W = terms_to_matrices(terms_orig)

    return {"U": U, "V": V, "W": W}
