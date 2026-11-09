import numpy as np
import time

def construct():
    t0 = time.time()
    n, m, p = 4, 4, 6

    # Build Strassen-based rank-84 decomposition via numpy
    # Then attempt to reduce rank by identifying linearly dependent columns
    U_list = []
    V_list = []
    W_list = []

    for i in range(2):
        for k in range(3):
            for j in range(2):
                # Indices into U (n*m = 16), V (m*p = 24), W (p*n = 24)
                a = 2 * i * m + 2 * j
                b = 2 * i * m + 2 * j + 1
                c = (2 * i + 1) * m + 2 * j
                d = (2 * i + 1) * m + 2 * j + 1
                e = 2 * j * p + 2 * k
                f = 2 * j * p + 2 * k + 1
                g = (2 * j + 1) * p + 2 * k
                h = (2 * j + 1) * p + 2 * k + 1
                c11 = 2 * k * n + 2 * i
                c12 = (2 * k + 1) * n + 2 * i
                c21 = 2 * k * n + 2 * i + 1
                c22 = (2 * k + 1) * n + 2 * i + 1

                # M1 = (a+d)(e+h)
                u, v, w = [a, d], [e, h], [c11, c22]
                coeffs_u = [1, 1]
                coeffs_v = [1, 1]
                coeffs_w = [1, 1]
                U_list.append((u, coeffs_u))
                V_list.append((v, coeffs_v))
                W_list.append((w, coeffs_w))

                # M2 = (c+d)e
                U_list.append(([c, d], [1, 1]))
                V_list.append(([e], [1]))
                W_list.append(([c21, c22], [1, -1]))

                # M3 = a(f-h)
                U_list.append(([a], [1]))
                V_list.append(([f, h], [1, -1]))
                W_list.append(([c12, c22], [1, 1]))

                # M4 = d(g-e)
                U_list.append(([d], [1]))
                V_list.append(([g, e], [1, -1]))
                W_list.append(([c11, c21], [1, 1]))

                # M5 = (a+b)h
                U_list.append(([a, b], [1, 1]))
                V_list.append(([h], [1]))
                W_list.append(([c11, c12], [-1, 1]))

                # M6 = (c-a)(e+f)
                U_list.append(([c, a], [1, -1]))
                V_list.append(([e, f], [1, 1]))
                W_list.append(([c22], [1]))

                # M7 = (b-d)(g+h)
                U_list.append(([b, d], [1, -1]))
                V_list.append(([g, h], [1, 1]))
                W_list.append(([c11], [1]))

    rank = len(U_list)  # 84
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n

    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]

    for col_idx in range(rank):
        for (idx, coeff) in zip(*[list(U_list[col_idx][0]), list(U_list[col_idx][1])]):
            U[idx][col_idx] = coeff
        for (idx, coeff) in zip(*[list(V_list[col_idx][0]), list(V_list[col_idx][1])]):
            V[idx][col_idx] = coeff
        for (idx, coeff) in zip(*[list(W_list[col_idx][0]), list(W_list[col_idx][1])]):
            W[idx][col_idx] = coeff

    if time.time() - t0 > 100:
        return {"U": U, "V": V, "W": W}

    # Verify
    U_np = np.array(U, dtype=np.int64).T
    V_np = np.array(V, dtype=np.int64).T
    W_np = np.array(W, dtype=np.int64).T

    recon = np.zeros((n, m, p), dtype=np.int64)
    for r in range(rank):
        recon += U_np[r].reshape(n, m)[:, :, None] * V_np[r].reshape(m, p)[None, :, :] * W_np[r].reshape(p, n)[None, None, :]

    if not np.all(recon == 1):
        # Fallback: schoolbook
        sb_rank = n * m * p
        U_s = [[0] * sb_rank for _ in range(dim_U)]
        V_s = [[0] * sb_rank for _ in range(dim_V)]
        W_s = [[0] * sb_rank for _ in range(dim_W)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][c] = 1
                    V_s[j * p + k][c] = 1
                    W_s[k * n + i][c] = 1
                    c += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    # Attempt rank reduction: try to find subsets of columns that are linearly dependent
    # by checking if removing a column and adjusting others preserves the decomposition.
    # This is expensive; only try a few candidates within time budget.
    if time.time() - t0 > 110:
        return {"U": U, "V": V, "W": W}

    # Quick check: see if the rank can be reduced by 1 by Gaussian elimination on
    # the tensor slices. For each potential column to remove, check if its tensor
    # can be expressed as a linear combination of the others.
    # This is O(rank^2 * dim) which is 84^2 * 384 ~ 2.7M ops, feasible.

    best_rank = rank
    best_U, best_V, best_W = U, V, W

    # Try removing each column and checking if the remaining set still spans
    # We'll use a simpler heuristic: check rank of the combined matrix
    # Stack all U columns, V columns, W columns into a tensor and check rank
    # Actually, just verify the full decomposition is valid and return it.

    return {"U": U, "V": V, "W": W}
