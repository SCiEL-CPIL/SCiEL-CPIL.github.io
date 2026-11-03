"""Refined block-Strassen decomposition with dependency removal + exact polish."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    import sys

    sys.setrecursionlimit(10000)
    n, m, p = DIMENSIONS  # 3, 4, 6
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit

    def elapsed():
        return time.time() - start_time

    # ---------- schoolbook fallback ----------
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def verify(U, V, W, r, tol=1e-9):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            T += u[:, :, None] * v[None, :, :] * w[None, None, :]
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    def make_builder(A_row_blocks):
        terms = []

        def add_term(U_entries, V_entries, W_entries):
            Ud = {}
            for (i, j, c) in U_entries:
                Ud[(i, j)] = Ud.get((i, j), 0) + c
            Vd = {}
            for (j, k, c) in V_entries:
                Vd[(j, k)] = Vd.get((j, k), 0) + c
            Wd = {}
            for (k, i, c) in W_entries:
                Wd[(k, i)] = Wd.get((k, i), 0) + c
            Ud = {k: v for k, v in Ud.items() if v != 0}
            Vd = {k: v for k, v in Vd.items() if v != 0}
            Wd = {k: v for k, v in Wd.items() if v != 0}
            if Ud and Vd and Wd:
                terms.append((Ud, Vd, Wd))

        def strassen_2x2(A_rows, A_cols, B_rows, B_cols):
            r0, r1 = A_rows
            c0, c1 = A_cols
            s0, s1 = B_rows
            t0, t1 = B_cols
            add_term([(r0, c0, 1), (r1, c1, 1)], [(s0, t0, 1), (s1, t1, 1)], [(t0, r0, 1), (t1, r1, 1)])
            add_term([(r1, c0, 1), (r1, c1, 1)], [(s0, t0, 1)], [(t0, r1, 1), (t1, r1, -1)])
            add_term([(r0, c0, 1)], [(s0, t1, 1), (s1, t1, -1)], [(t1, r0, 1), (t1, r1, 1)])
            add_term([(r1, c1, 1)], [(s1, t0, 1), (s0, t0, -1)], [(t0, r1, 1), (t0, r0, 1)])
            add_term([(r0, c0, 1), (r0, c1, 1)], [(s1, t1, 1)], [(t1, r1, 1), (t0, r0, -1)])
            add_term([(r1, c1, 1), (r0, c1, -1)], [(s0, t0, 1), (s0, t1, 1)], [(t1, r1, 1)])
            add_term([(r0, c1, 1), (r1, c1, -1)], [(s1, t1, 1), (s0, t1, 1)], [(t0, r0, 1)])

        def direct_block(A_rows, A_cols, B_rows, B_cols):
            for i in A_rows:
                for j in A_cols:
                    for k in B_rows:
                        if j == k:
                            for l in B_cols:
                                add_term([(i, j, 1)], [(k, l, 1)], [(l, i, 1)])

        col_blocks = [(0, 1), (2, 3)]
        C_col_blocks = [(0, 1, 2), (3, 4, 5)]

        for A_rb in A_row_blocks:
            for cb in col_blocks:
                for B_rb in col_blocks:
                    for C_cb in C_col_blocks:
                        if len(A_rb) == 2 and len(cb) == 2 and len(B_rb) == 2 and len(C_cb) == 2:
                            strassen_2x2(A_rb, cb, B_rb, C_cb)
                        else:
                            direct_block(A_rb, cb, B_rb, C_cb)
        return terms

    def build_matrices(terms):
        r = len(terms)
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        for c, (Ud, Vd, Wd) in enumerate(terms):
            for (i, j), coeff in Ud.items():
                U[i * m + j, c] = coeff
            for (j, k), coeff in Vd.items():
                V[j * p + k, c] = coeff
            for (k, i), coeff in Wd.items():
                W[k * n + i, c] = coeff
        return U, V, W, r

    def remove_redundant(terms):
        seen = {}
        kept = []
        for Ud, Vd, Wd in terms:
            allvals = list(Ud.values()) + list(Vd.values()) + list(Wd.values())
            mabs = max(abs(v) for v in allvals)
            if mabs == 0:
                continue
            scale = 1.0 / mabs
            key = (
                frozenset((k, round(v * scale, 9)) for k, v in Ud.items()),
                frozenset((k, round(v * scale, 9)) for k, v in Vd.items()),
                frozenset((k, round(v * scale, 9)) for k, v in Wd.items()),
            )
            if key in seen:
                prev = seen[key]
                common = None
                for kk in Ud:
                    if kk in prev[0]:
                        common = kk
                        break
                if common is not None and Ud[common] * prev[0][common] < 0:
                    kept = [t for t in kept if t is not prev]
                    del seen[key]
                    continue
                continue
            seen[key] = (Ud, Vd, Wd)
            kept.append((Ud, Vd, Wd))
        return kept

    def to_half_integer(arr):
        return np.round(np.asarray(arr, dtype=float) * 2) / 2.0

    def final_check_and_fix(U, V, W, r):
        if verify(U, V, W, r):
            return U, V, W, r
        Uh, Vh, Wh = to_half_integer(U), to_half_integer(V), to_half_integer(W)
        if verify(Uh, Vh, Wh, r):
            return Uh, Vh, Wh, r
        return U_school, V_school, W_school, rank_school

    best = (U_school, V_school, W_school, rank_school)

    # primary partition: rows (0,1)+(2), cols (0,1)+(2,3), Ccols (0,1,2)+(3,4,5)
    if elapsed() < deadline - 30:
        terms = make_builder([(0, 1), (2,)])
        terms = remove_redundant(terms)
        U, V, W, r = build_matrices(terms)
        if verify(U, V, W, r):
            U, V, W, r = final_check_and_fix(U, V, W, r)
            if r < best[3]:
                best = (U, V, W, r)

    # alternate row partitions
    if elapsed() < deadline - 15:
        for rb in [([(0,), (1, 2)]), ([(0, 1, 2)]), ([(0,), (1,), (2,)])]:
            if elapsed() > deadline - 10:
                break
            try:
                terms2 = make_builder(rb)
            except Exception:
                continue
            terms2 = remove_redundant(terms2)
            U2, V2, W2, r2 = build_matrices(terms2)
            if verify(U2, V2, W2, r2):
                U2, V2, W2, r2 = final_check_and_fix(U2, V2, W2, r2)
                if r2 < best[3]:
                    best = (U2, V2, W2, r2)

    U_out, V_out, W_out, r_out = best
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
