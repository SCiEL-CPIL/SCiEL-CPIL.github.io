import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = 135

    n, m, p = DIMENSIONS

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Hybrid: Strassen for first 2 cols, Schoolbook for rest
    if p >= 2:
        strassen_rank = 7
        schoolbook_cols = p - 2
        schoolbook_rank = 4 * schoolbook_cols
        total_rank = strassen_rank + schoolbook_rank

        if total_rank < best_rank:
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]

            col = 0
            # Strassen terms for A(2x2) * B(2x2) = C(2x2)
            # Indices:
            # U: 0=a, 1=b, 2=c, 3=d
            # V: 0=e, 1=f, 13=g, 14=h
            # W: 0=C00, 1=C10, 2=C01, 3=C11
            
            # Correct Strassen terms
            strassen_terms = [
                # M1 = (a+b)(e+f)
                ([1, 1, 0, 0], {0: 1, 1: 1}, {0: 1, 3: 1}),
                # M2 = (c+d)e
                ([0, 0, 1, 1], {0: 1}, {1: 1, 3: -1}),
                # M3 = a(h-f)
                ([1, 0, 0, 0], {14: 1, 1: -1}, {2: 1}),
                # M4 = d(g-e)
                ([0, 0, 0, 1], {13: 1, 0: -1}, {0: 1}),
                # M5 = (a+d)(e+h)
                ([1, 0, 0, 1], {0: 1, 14: 1}, {2: 1, 3: 1}),
                # M6 = (b+c)(g-e)
                ([0, 1, 1, 0], {13: 1, 0: -1}, {1: 1}),
                # M7 = (b+d)(f-h)
                ([0, 1, 0, 1], {1: 1, 14: -1}, {0: -1}),
            ]

            for U_vec, V_dict, W_dict in strassen_terms:
                for i in range(4):
                    if U_vec[i] != 0:
                        U[i][col] = U_vec[i]
                for j_idx, val in V_dict.items():
                    V[j_idx][col] = val
                for k_idx, val in W_dict.items():
                    W[k_idx][col] = val
                col += 1

            # Schoolbook for columns 2..12
            for i in range(n):
                for j in range(m):
                    for k in range(2, p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1

            # Verify
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)

            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test

            C_decomp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for t in range(total_rank):
                            C_decomp[i, k] += U_np[i*m+j, t] * V_np[j*p+k, t] * W_np[k*n+i, t]
            
            if np.allclose(C_decomp, C_expected, atol=1e-10):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = total_rank

            if time.time() - start_time > deadline - 5:
                return best_result

    return best_result
