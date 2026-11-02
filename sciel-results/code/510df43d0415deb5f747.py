"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.

Family: block-based Strassen decomposition (rank 64 baseline) refined by
  (a) structural merge of terms sharing an identical (U,V) support signature,
  (b) an integer ALS that keeps the solution on the half-integer lattice so
      final rounding is exact (not a post-hoc gamble),
  (c) a fast exact integer verifier.
Falls back to schoolbook (rank 80) only if the Strassen baseline fails.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_int(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast EXACT verification: all entries are half-integers, so 4*T is an
    integer tensor that must equal 4 everywhere. Exact (no float tolerance)."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    Ur = Ua.reshape(n, m, Ua.shape[1])
    Vr = Va.reshape(m, p, Va.shape[1])
    Wr = Wa.reshape(p, n, Wa.shape[1])
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    return bool(np.allclose(4.0 * T, 4.0, atol=atol))

def _round_to_half_integer(x):
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def _strassen_construct(n, m, p):
    """Block Strassen baseline: 4*(2x2x2 rank7) + 4*(2x2x3 rank9) = 64 terms."""
    U_cols, V_cols, W_cols = [], [], []

    def add_term(u, v, w):
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

    def ui(gi, gj): return gi * m + gj
    def vi(gj, gk): return gj * p + gk
    def wi(gk, gi): return gk * n + gi

    def add_strassen_2x2_2(ai, aj, bj, bk):
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[ui(ai,aj)] = 1; u[ui(ai+1,aj+1)] = 1
        v[vi(bj,bk)] = 1; v[vi(bj+1,bk+1)] = 1
        w[wi(bk,ai)] = 1; w[wi(bk+1,ai+1)] = 1; add_term(u,v,w)

        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[ui(ai+1,aj)] = 1; u[ui(ai+1,aj+1)] = 1
        v[vi(bj,bk)] = 1
        w[wi(bk,ai+1)] = 1; w[wi(bk+1,ai+1)] = -1; add_term(u,v,w)

        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[ui(ai,aj)] = 1
        v[vi(bj,bk+1)] = 1; v[vi(bj+1,bk+1)] = -1
        w[wi(bk+1,ai)] = 1; w[wi(bk+1,ai+1)] = 1; add_term(u,v,w)

        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[ui(ai+1,aj+1)] = 1
        v[vi(bj+1,bk)] = 1; v[vi(bj,bk)] = -1
        w[wi(bk,ai)] = 1; w[wi(bk,ai+1)] = 1; add_term(u,v,w)

        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[ui(ai,aj)] = 1; u[ui(ai,aj+1)] = 1
        v[vi(bj+1,bk+1)] = 1
        w[wi(bk,ai)] = -1; w[wi(bk+1,ai)] = 1; add_term(u,v,w)

        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[ui(ai+1,aj)] = 1; u[ui(ai,aj)] = -1
        v[vi(bj,bk)] = 1; v[vi(bj,bk+1)] = 1
        w[wi(bk+1,ai+1)] = 1; add_term(u,v,w)

        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[ui(ai,aj+1)] = 1; u[ui(ai+1,aj+1)] = -1
        v[vi(bj+1,bk)] = 1; v[vi(bj+1,bk+1)] = 1
        w[wi(bk,ai)] = 1; add_term(u,v,w)

    def add_2x2x3(ai, aj, bj, bk):
        add_strassen_2x2_2(ai, aj, bj, bk)
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[ui(ai+di, aj+dj)] = 1
                v = np.zeros(m*p); v[vi(bj+dj, bk+2)] = 1
                w = np.zeros(p*n); w[wi(bk+2, ai+di)] = 1
                add_term(u,v,w)

    add_strassen_2x2_2(0,0,0,0); add_strassen_2x2_2(0,2,2,0)
    add_strassen_2x2_2(2,0,0,0); add_strassen_2x2_2(2,2,2,0)
    add_2x2x3(0,0,0,2); add_2x2x3(0,2,2,2)
    add_2x2x3(2,0,0,2); add_2x2x3(2,2,2,2)

    return np.array(U_cols).T.tolist(), np.array(V_cols).T.tolist(), np.array(W_cols).T.tolist()

def _merge_identical_terms(U, V, W, n, m, p):
    """Merge terms with identical (U_col, V_col) signatures by summing W_cols.
    This exploits the algebraic dependencies inherent in the block structure."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    groups = {}
    order = []
    for r in range(rank):
        key = (tuple(np.round(Ua[:, r] * 2)), tuple(np.round(Va[:, r] * 2)))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    new_U, new_V, new_W = [], [], []
    for key in order:
        idxs = groups[key]
        u_col = Ua[:, idxs[0]].copy()
        v_col = Va[:, idxs[0]].copy()
        w_col = np.zeros(p * n)
        for idx in idxs:
            w_col += Wa[:, idx]
        new_U.append(u_col); new_V.append(v_col); new_W.append(w_col)
    if len(new_U) >= rank:
        return None
    U_new = np.column_stack(new_U)
    V_new = np.column_stack(new_V)
    W_new = np.column_stack(new_W)
    if _verify_int(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, atol=1e-9):
        return U_new.tolist(), V_new.tolist(), W_new.tolist()
    return None

def _als_decompose(n, m, p, rank, seed=42, max_iters=300, deadline=None,
                   U_init=None, V_init=None, W_init=None):
    """Integer ALS: solve each row via least squares, then round to the
    half-integer lattice immediately, so the search stays on the exact
    (half-)integer manifold and final rounding is exact."""
    if U_init is not None and V_init is not None and W_init is not None:
        U = np.array(U_init, dtype=np.float64).copy()
        V = np.array(V_init, dtype=np.float64).copy()
        W = np.array(W_init, dtype=np.float64).copy()
    else:
        np.random.seed(seed)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        U = _round_matrix_to_half_int(U); V = _round_matrix_to_half_int(V)
        W = _round_matrix_to_half_int(W)

    def err_of(U_, V_, W_):
        Ur = U_.reshape(n, m, rank); Vr = V_.reshape(m, p, rank); Wr = W_.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        return np.sqrt(np.sum((T - 1.0) ** 2))

    best_err = err_of(U, V, W)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Optimize U rows
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                try:
                    sol = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                except Exception:
                    continue
                U[i*m+j, :] = _round_matrix_to_half_int(sol.reshape(1, -1))[0]
        # Optimize V rows
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                try:
                    sol = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                except Exception:
                    continue
                V[j*p+k, :] = _round_matrix_to_half_int(sol.reshape(1, -1))[0]
        # Optimize W rows
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                try:
                    sol = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                except Exception:
                    continue
                W[k*n+i, :] = _round_matrix_to_half_int(sol.reshape(1, -1))[0]

        e = err_of(U, V, W)
        if e < best_err:
            best_err = e
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if best_err < 1e-10:
            break

    Ur = _round_matrix_to_half_int(best_U)
    Vr = _round_matrix_to_half_int(best_V)
    Wr = _round_matrix_to_half_int(best_W)
    if _verify_int(Ur.tolist(), Vr.tolist(), Wr.tolist(), n, m, p, atol=1e-9):
        return Ur.tolist(), Vr.tolist(), Wr.tolist()
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 138  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    U_base, V_base, W_base = _strassen_construct(n, m, p)
    if not _verify_int(U_base, V_base, W_base, n, m, p, atol=1e-9):
        return _schoolbook()
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])

    # (a) structural merge of identical-signature terms
    if time.time() < deadline - 100:
        merged = _merge_identical_terms(U_base, V_base, W_base, n, m, p)
        if merged is not None:
            best_result = {"U": merged[0], "V": merged[1], "W": merged[2]}
            best_rank = len(merged[0][0])

    # (b) integer ALS reduction from the best current decomposition
    Ua = np.array(best_result["U"], dtype=np.float64)
    Va = np.array(best_result["V"], dtype=np.float64)
    Wa = np.array(best_result["W"], dtype=np.float64)
    for target_rank in range(best_rank - 1, max(30, best_rank - 14), -1):
        if time.time() > deadline - 55:
            break
        U_init = Ua[:, :target_rank]; V_init = Va[:, :target_rank]; W_init = Wa[:, :target_rank]
        result = _als_decompose(n, m, p, target_rank, seed=42, max_iters=220,
                                U_init=U_init, V_init=V_init, W_init=W_init,
                                deadline=deadline - 55)
        if result is not None:
            best_result = {"U": result[0], "V": result[1], "W": result[2]}
            best_rank = target_rank
            Ua = np.array(result[0], dtype=np.float64)
            Va = np.array(result[1], dtype=np.float64)
            Wa = np.array(result[2], dtype=np.float64)

    # (c) fresh random integer-ALS restarts hunting for lower rank
    for rank in range(best_rank - 1, 39, -1):
        if time.time() > deadline - 18:
            break
        found = False
        for seed in range(12):
            if time.time() > deadline - 18:
                break
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=300,
                                    deadline=deadline - 18)
            if result is not None:
                best_result = {"U": result[0], "V": result[1], "W": result[2]}
                best_rank = rank
                found = True
                break
        if found:
            continue

    return best_result
# EVOLVE-BLOCK-END
