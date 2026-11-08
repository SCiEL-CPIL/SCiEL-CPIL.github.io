"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Builds a Strassen+direct base decomposition (rank 68), then performs fast,
correct iterative rank reduction by checking (with 72-dim vectorization)
whether each term's tensor-vector lies in the span of the others.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    # Precompute the list of active (i, j, k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    ntrip = len(triples)  # 72

    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)

        U_t.append([(a, 1), (d, 1)]);  V_t.append([(e, 1), (h, 1)]);      W_t.append([(C00, 1), (C11, 1)])
        U_t.append([(c, 1), (d, 1)]);  V_t.append([(e, 1)]);              W_t.append([(C10, 1), (C11, -1)])
        U_t.append([(a, 1)]);          V_t.append([(f, 1), (h, -1)]);     W_t.append([(C01, 1), (C11, 1)])
        U_t.append([(d, 1)]);          V_t.append([(g, 1), (e, -1)]);     W_t.append([(C00, 1), (C10, 1)])
        U_t.append([(a, 1), (b, 1)]);  V_t.append([(h, 1)]);              W_t.append([(C00, -1), (C01, 1)])
        U_t.append([(c, 1), (a, -1)]); V_t.append([(e, 1), (f, 1)]);      W_t.append([(C11, 1)])
        U_t.append([(b, 1), (d, -1)]); V_t.append([(g, 1), (h, 1)]);      W_t.append([(C00, 1)])

    def direct_term(i, j, k, U_t, V_t, W_t):
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])

    def build_base():
        U_t, V_t, W_t = [], [], []
        for k_off in range(0, p, 2):
            strassen_decomp(0, 0, 0, k_off, U_t, V_t, W_t)
        for i in range(2):
            for k in range(p):
                direct_term(i, 2, k, U_t, V_t, W_t)
        for j in range(2):
            for k in range(p):
                direct_term(2, j, k, U_t, V_t, W_t)
        for k in range(p):
            direct_term(2, 2, k, U_t, V_t, W_t)
        return U_t, V_t, W_t

    def to_matrices(U_t, V_t, W_t, rank):
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for idx in range(rank):
            for (i, j), coeff in U_t[idx]:
                U[i * m + j, idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V[j * p + k, idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W[k * n + i, idx] = coeff
        return U, V, W

    def verify_decomp(Ua, Va, Wa, nn, mm, pp):
        # Vectorized check over the 72 active triples
        for (i, j, k) in triples:
            s = np.sum(Ua[i * mm + j] * Va[j * pp + k] * Wa[k * nn + i])
            if abs(s - 1.0) > 1e-7:
                return False
        return True

    def tensor_vec(U, V, W, r):
        """72-dim vector: entry s = U[i,j,r]*V[j,k,r]*W[k,i,r] for triple s."""
        out = np.empty(ntrip, dtype=float)
        for s, (i, j, k) in enumerate(triples):
            out[s] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        return out

    # Build base
    U_t, V_t, W_t = build_base()
    rank = len(W_t)
    U, V, W = to_matrices(U_t, V_t, W_t, rank)

    def to_nice_list(arr):
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(float(val))
            result.append(nice_row)
        return result

    # Fallback schoolbook
    def schoolbook():
        R = n * m * p
        Uf = np.zeros((n * m, R), dtype=float)
        Vf = np.zeros((m * p, R), dtype=float)
        Wf = np.zeros((p * n, R), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i * m + j, col] = 1
                    Vf[j * p + k, col] = 1
                    Wf[k * n + i, col] = 1
                    col += 1
        return Uf, Vf, Wf

    if not verify_decomp(U, V, W, n, m, p):
        Uf, Vf, Wf = schoolbook()
        return {"U": to_nice_list(Uf), "V": to_nice_list(Vf), "W": to_nice_list(Wf)}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank

    # FAST iterative rank reduction with correct 72-dim vectorization
    def try_reduce(Uc, Vc, Wc, max_rounds=200):
        """Greedy peeling: repeatedly remove a term whose tensor-vec is in span of others."""
        rank_c = Wc.shape[1]
        progress = True
        rounds = 0
        while progress and rounds < max_rounds:
            progress = False
            rounds += 1
            if time.time() - start_time > DEADLINE:
                break
            # Build the 72 x rank_c matrix of tensor vectors
            M = np.empty((ntrip, rank_c), dtype=float)
            for r in range(rank_c):
                M[:, r] = tensor_vec(Uc, Vc, Wc, r)
            # For each r, check if M[:,r] in span of M[:, :r] + M[:, r+1:]
            # Use: rank of M == rank of M without column r  =>  column r dependent
            # Efficient: compute once, then for each r use the SVD / QR
            # Simpler: for each r, lstsq the others against M[:,r]
            # To speed up, only check if column r has any nonzero entries
            for r in range(rank_c - 1, -1, -1):
                if time.time() - start_time > DEADLINE:
                    return Uc, Vc, Wc
                if np.all(np.abs(M[:, r]) < 1e-12):
                    # zero column -> remove
                    Uc = np.delete(Uc, r, axis=1)
                    Vc = np.delete(Vc, r, axis=1)
                    Wc = np.delete(Wc, r, axis=1)
                    M = np.delete(M, r, axis=1)
                    rank_c -= 1
                    progress = True
                    continue
                # check dependence via lstsq on others
                others = np.delete(M, r, axis=1)
                try:
                    coeffs, res, rk, _ = np.linalg.lstsq(others, M[:, r], rcond=1e-10)
                    recon = others @ coeffs
                    if np.allclose(recon, M[:, r], atol=1e-8, rtol=1e-8):
                        Uc = np.delete(Uc, r, axis=1)
                        Vc = np.delete(Vc, r, axis=1)
                        Wc = np.delete(Wc, r, axis=1)
                        M = np.delete(M, r, axis=1)
                        rank_c -= 1
                        progress = True
                except np.linalg.LinAlgError:
                    pass
        return Uc, Vc, Wc

    U, V, W = try_reduce(U, V, W)
    if verify_decomp(U, V, W, n, m, p):
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = W.shape[1]

    # Second pass: try with random term scalings to break symmetry, in case
    # the greedy order missed dependencies. (Cheap: scaling doesn't change tensor.)
    if time.time() - start_time < DEADLINE - 10:
        rng = np.random.default_rng(42)
        for trial in range(8):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Start from the best so far
            Uc, Vc, Wc = best_U.copy(), best_V.copy(), best_W.copy()
            # Apply random scalings: u_r *= a, v_r *= b, w_r *= 1/(a*b)
            R0 = Wc.shape[1]
            a = rng.uniform(0.5, 2.0, size=R0)
            b = rng.uniform(0.5, 2.0, size=R0)
            c = 1.0 / (a * b)
            Uc = Uc * a[None, :]
            Vc = Vc * b[None, :]
            Wc = Wc * c[None, :]
            Uc, Vc, Wc = try_reduce(Uc, Vc, Wc, max_rounds=100)
            if verify_decomp(Uc, Vc, Wc, n, m, p):
                if Wc.shape[1] < best_rank:
                    best_rank = Wc.shape[1]
                    best_U, best_V, best_W = Uc.copy(), Vc.copy(), Wc.copy()

    # Final verification; fallback if broken
    if not verify_decomp(best_U, best_V, best_W, n, m, p):
        # Rebuild base and reduce once more deterministically
        U2, V2, W2 = to_matrices(*build_base(), len(build_base()[2]))
        # rebuild properly
        Ut2, Vt2, Wt2 = build_base()
        U2, V2, W2 = to_matrices(Ut2, Vt2, Wt2, len(Wt2))
        U2, V2, W2 = try_reduce(U2, V2, W2)
        if verify_decomp(U2, V2, W2, n, m, p):
            best_U, best_V, best_W = U2, V2, W2
        else:
            Uf, Vf, Wf = schoolbook()
            best_U, best_V, best_W = Uf, Vf, Wf

    return {"U": to_nice_list(best_U), "V": to_nice_list(best_V), "W": to_nice_list(best_W)}
# EVOLVE-BLOCK-END
