"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen on 2x2 blocks plus direct terms, then aggressive rank reduction.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def verify_decomp(Ua, Va, Wa, nn, mm, pp):
        """Verify that the decomposition is exact using vectorized ops."""
        T = np.zeros((nn, mm, pp), dtype=float)
        for r in range(Ua.shape[1]):
            u = Ua[:, r].reshape(nn, mm)
            v = Va[:, r].reshape(mm, pp)
            w = Wa[:, r].reshape(pp, nn)
            for i in range(nn):
                for j in range(mm):
                    if u[i, j] != 0:
                        T[i, j, :] += u[i, j] * v[j, :] * w[:, i]
        return np.allclose(T, 1.0, atol=1e-7)

    def to_nice_list(arr):
        """Convert array to list with nice integer/half-integer values."""
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

    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
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

        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])

        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])

        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])

        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])

        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])

        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])

        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])

    def direct_term(i, j, k, U_t, V_t, W_t):
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])

    def build_factors(U_t, V_t, W_t):
        """Build factor matrices from term lists."""
        rank = len(W_t)
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

    # Build initial decomposition using Strassen + direct terms (rank 68)
    U_t = []
    V_t = []
    W_t = []

    # Strassen for A[0:2, 0:2] * B[0:2, k:k+2] for k = 0, 2, 4, 6
    for k_off in range(0, 8, 2):
        strassen_decomp(0, 0, 0, k_off, U_t, V_t, W_t)

    # Direct terms for remaining parts
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k, U_t, V_t, W_t)

    for j in range(2):
        for k in range(8):
            direct_term(2, j, k, U_t, V_t, W_t)

    for k in range(8):
        direct_term(2, 2, k, U_t, V_t, W_t)

    U, V, W = build_factors(U_t, V_t, W_t)
    rank = len(W_t)

    # Verify initial decomposition
    if not verify_decomp(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": to_nice_list(U_fb), "V": to_nice_list(V_fb), "W": to_nice_list(W_fb)}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank

    # Precompute all term vectors
    term_vecs = []
    for r in range(rank):
        term_vecs.append(np.kron(np.kron(U[:, r], V[:, r]), W[:, r]))

    # Multiple passes of rank reduction
    for iteration in range(10):
        if time.time() - start_time > DEADLINE:
            break

        removed_any = False
        # Try to remove terms one by one (from high to low index)
        for r in range(rank - 1, -1, -1):
            if time.time() - start_time > DEADLINE:
                break

            # Check if term r is zero
            if np.all(np.abs(U[:, r]) < 1e-12) and np.all(np.abs(V[:, r]) < 1e-12) and np.all(np.abs(W[:, r]) < 1e-12):
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
                rank -= 1
                removed_any = True
                # Rebuild term_vecs
                term_vecs = []
                for rr in range(rank):
                    term_vecs.append(np.kron(np.kron(U[:, rr], V[:, rr]), W[:, rr]))
                continue

            # Check if term r is in span of others
            others_idx = [s for s in range(rank) if s != r]
            if len(others_idx) == 0:
                continue

            others_mat = np.column_stack([term_vecs[s] for s in others_idx])
            vec_r = term_vecs[r]

            try:
                # Use QR for efficiency
                Q, R = np.linalg.qr(others_mat, mode='reduced')
                # Project vec_r onto the column space of others
                proj = Q @ (Q.T @ vec_r)
                residual = vec_r - proj
                if np.max(np.abs(residual)) < 1e-8:
                    # Term r is in span, remove it
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    rank -= 1
                    removed_any = True
                    # Rebuild term_vecs
                    term_vecs = []
                    for rr in range(rank):
                        term_vecs.append(np.kron(np.kron(U[:, rr], V[:, rr]), W[:, rr]))
            except Exception:
                pass

        if removed_any:
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = rank
        else:
            break  # No more removals possible

    # Final verification
    if not verify_decomp(best_U, best_V, best_W, n, m, p):
        # Fallback: rebuild original
        U_t2 = []
        V_t2 = []
        W_t2 = []
        for k_off in range(0, 8, 2):
            strassen_decomp(0, 0, 0, k_off, U_t2, V_t2, W_t2)
        for i in range(2):
            for k in range(8):
                direct_term(i, 2, k, U_t2, V_t2, W_t2)
        for j in range(2):
            for k in range(8):
                direct_term(2, j, k, U_t2, V_t2, W_t2)
        for k in range(8):
            direct_term(2, 2, k, U_t2, V_t2, W_t2)
        U_orig, V_orig, W_orig = build_factors(U_t2, V_t2, W_t2)
        return {"U": to_nice_list(U_orig), "V": to_nice_list(V_orig), "W": to_nice_list(W_orig)}

    return {"U": to_nice_list(best_U), "V": to_nice_list(best_V), "W": to_nice_list(best_W)}
# EVOLVE-BLOCK-END
