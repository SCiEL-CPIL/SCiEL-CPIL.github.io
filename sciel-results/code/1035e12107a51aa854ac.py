import numpy as np
import time

def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    def add_strassen_222(U, V, W, i0, j0, k0, col):
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11, col] += 1
        col += 1
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00, col] += 1
        col += 1
        return col

    def add_schoolbook_221(U, V, W, i0, j0, k0, col):
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10, col] += 1; col += 1
        return col

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Baseline: 2 j-pairs, each with three 2x2x2 Strassen + one 2x2x1 schoolbook -> rank 50
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for js in [0, 2]:
        for ks in [0, 2, 4]:
            col = add_strassen_222(U, V, W, 0, js, ks, col)
        col = add_schoolbook_221(U, V, W, 0, js, 6, col)
    assert col == 50
    assert verify(U, V, W, 1e-9), "baseline rank-50 decomposition failed verification"

    best_U, best_V, best_W = U, V, W
    best_rank = rank

    # ---- Rank reduction pass 1: drop provably-redundant (all-zero) rank columns ----
    def drop_zero_cols(U, V, W):
        r = U.shape[1]
        keep = [c for c in range(r)
                if np.any(U[:, c] != 0) and np.any(V[:, c] != 0) and np.any(W[:, c] != 0)]
        if len(keep) == r:
            return U, V, W
        return U[:, keep].copy(), V[:, keep].copy(), W[:, keep].copy()

    U, V, W = drop_zero_cols(U, V, W)
    if verify(U, V, W, 1e-9):
        best_U, best_V, best_W = U, V, W
        best_rank = U.shape[1]

    # ---- Rank reduction pass 2: "absorption" polish ----
    # Hypothesis: some rank column t can be expressed as a linear combination of the
    # others such that the tensor is preserved (a smooth/epigraph-style local solve).
    # For each candidate column t, solve a linear system over the remaining columns and
    # keep the drop only if the full tensor still verifies exactly.
    def try_absorb(U, V, W, t):
        r = U.shape[1]
        others = [c for c in range(r) if c != t]
        if len(others) < 1:
            return None
        # Unknowns x_c (c in others) minimizing the residual of the dropped column's
        # contribution. We form a linear system: for each (i,j,k), the dropped column
        # contributes U_t[i,j]*V_t[j,k]*W_t[k,i]. We want sum_c x_c * U_c[i,j]*V_c[j,k]*W_c[k,i] = that.
        # Build rows over (i,j,k) and columns over c.
        rows = []
        rhs = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = np.array([U[o, i * m + j] * V[o, j * p + k] * W[o, k * n + i] for o in others])
                    rows.append(row)
                    rhs.append(U[t, i * m + j] * V[t, j * p + k] * W[t, k * n + i])
        A_mat = np.array(rows)
        b_vec = np.array(rhs)
        try:
            x, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except Exception:
            return None
        # Build candidate without column t, adding x_c * col_c into the others
        U2 = U[:, others].copy()
        V2 = V[:, others].copy()
        W2 = W[:, others].copy()
        for idx, o in enumerate(others):
            U2[:, idx] += x[idx] * U[:, o]
            V2[:, idx] += x[idx] * V[:, o]
            W2[:, idx] += x[idx] * W[:, o]
        return U2, V2, W2

    # Attempt absorption in a few orderings (highest-spread columns first is a heuristic)
    if time.time() < deadline - 15:
        r = U.shape[1]
        # heuristic order: columns with larger L2 norm first (more "spread", more likely absorbable)
        order = sorted(range(r), key=lambda c: -(np.linalg.norm(U[:, c]) + np.linalg.norm(V[:, c]) + np.linalg.norm(W[:, c])))
        for t in order:
            if time.time() > deadline - 10:
                break
            cand = try_absorb(U, V, W, t)
            if cand is None:
                continue
            Uc, Vc, Wc = cand
            if verify(Uc, Vc, Wc, 1e-8):
                U, V, W = Uc, Vc, Wc
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
                r = U.shape[1]
                # recompute order after a successful drop
                order = sorted(range(r), key=lambda c: -(np.linalg.norm(U[:, c]) + np.linalg.norm(V[:, c]) + np.linalg.norm(W[:, c])))

    # Final safety: guarantee a valid result
    if not verify(best_U, best_V, best_W, 1e-8):
        best_U, best_V, best_W = schoolbook()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
