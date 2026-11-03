"""Matrix multiplication 2x4 x 4x7: block Strassen + aggressive exact rank reduction."""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 4, 7]
DEADLINE = time.time() + 145.0


def _time_left():
    return DEADLINE - time.time()


def _get_strassen_222():
    r = 7
    U = np.zeros((4, r)); V = np.zeros((4, r)); W = np.zeros((4, r))
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W


def _get_school_221():
    r = 4
    U = np.zeros((4, r)); V = np.zeros((2, r)); W = np.zeros((2, r))
    for c in range(r):
        i = c // 2
        j = c % 2
        U[i * 2 + j, c] = 1.0
        V[j, c] = 1.0
        W[i, c] = 1.0
    return U, V, W


def _build_block_decomposition():
    n, m, p = 2, 4, 7
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    U_s, V_s, W_s = _get_strassen_222()
    U_c, V_c, W_c = _get_school_221()

    col = 0
    for j_off in (0, 2):
        for k_off in (0, 2, 4):
            r_sub = 7
            for i in range(2):
                for js in range(2):
                    U[i * m + (j_off + js), col:col + r_sub] = U_s[i * 2 + js, :]
            for js in range(2):
                for ks in range(2):
                    V[(j_off + js) * p + (k_off + ks), col:col + r_sub] = V_s[js * 2 + ks, :]
            for ks in range(2):
                for i in range(2):
                    W[(k_off + ks) * n + i, col:col + r_sub] = W_s[ks * 2 + i, :]
            col += r_sub
        r_sub = 4
        for i in range(2):
            for js in range(2):
                U[i * m + (j_off + js), col:col + r_sub] = U_c[i * 2 + js, :]
        for js in range(2):
            V[(j_off + js) * p + 6, col:col + r_sub] = V_c[js, :]
        for i in range(2):
            W[6 * n + i, col:col + r_sub] = W_c[i, :]
        col += r_sub
    return U, V, W


def _build_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _tensor_from(U, V, W):
    return np.einsum('ar,br,cr->abc', U, V, W)


def _verify(U, V, W, T, tol=1e-7):
    R = _tensor_from(U, V, W)
    return bool(np.allclose(R, T, atol=tol, rtol=0.0))


def _coeff_matrix(U, V, W, a, b, c):
    r = U.shape[1]
    C = np.empty((a * b * c, r), dtype=U.dtype)
    idx = 0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                C[idx, :] = U[i, :] * V[j, :] * W[k, :]
                idx += 1
    return C


def _round_half(M):
    return np.round(M * 2.0) / 2.0


def _is_half_integer(M, tol=1e-8):
    return bool(np.all(np.abs(M * 2.0 - np.round(M * 2.0)) < tol))


def _exact_linear_solve(C, T):
    """Exact rational Gaussian elimination. C: (N,r) float/int, T: (N,) float/int.
    Returns list of Fractions (solution) or None if no exact solution exists."""
    N, r = C.shape
    tol = 1e-9
    rows = []
    for i in range(N):
        row = []
        for j in range(r):
            v = C[i, j]
            if abs(v - round(v)) < tol:
                row.append(Fraction(int(round(v))))
            else:
                row.append(Fraction(v).limit_denominator(10**6))
        tv = T[i]
        if abs(tv - round(tv)) < tol:
            tval = Fraction(int(round(tv)))
        else:
            tval = Fraction(tv).limit_denominator(10**6)
        rows.append((row, tval))

    pivot_cols = []
    row_idx = 0
    for col in range(r):
        piv = None
        for rr in range(row_idx, N):
            if rows[rr][0][col] != 0:
                piv = rr
                break
        if piv is None:
            continue
        rows[row_idx], rows[piv] = rows[piv], rows[row_idx]
        pv = rows[row_idx][0][col]
        prow = rows[row_idx][0][:]
        ptval = rows[row_idx][1]
        for cc in range(col, r):
            prow[cc] = prow[cc] / pv
        ptval = ptval / pv
        rows[row_idx] = (prow, ptval)
        for rr in range(row_idx + 1, N):
            f = rows[rr][0][col]
            if f != 0:
                rrow = rows[rr][0][:]
                rtval = rows[rr][1]
                for cc in range(col, r):
                    rrow[cc] = rrow[cc] - f * prow[cc]
                rtval = rtval - f * ptval
                rows[rr] = (rrow, rtval)
        pivot_cols.append(col)
        row_idx += 1

    # Check consistency: any remaining nonzero row with zero lhs but nonzero rhs
    for rr in range(row_idx, N):
        if any(x != 0 for x in rows[rr][0]):
            continue
        if rows[rr][1] != 0:
            return None

    # Back-substitute
    sol = [Fraction(0)] * r
    for pi in range(row_idx - 1, -1, -1):
        pc = pivot_cols[pi]
        val = rows[pi][1]
        for cc in range(pc + 1, r):
            val -= rows[pi][0][cc] * sol[cc]
        sol[pc] = val
    return sol


def _try_drop_exact(U, V, W, T, a, b, c, drop):
    """Try to drop column 'drop' by solving exact linear system for remaining cols."""
    r = U.shape[1]
    keep = [d for d in range(r) if d != drop]
    U_k = U[:, keep]
    V_k = V[:, keep]
    W_k = W[:, keep]
    C_k = _coeff_matrix(U_k, V_k, W_k, a, b, c)
    T_flat = T.reshape(-1)
    sol = _exact_linear_solve(C_k, T_flat)
    if sol is None:
        return None
    # Check sol is half-integer (so we can keep clean entries)
    sol_ok = True
    for s in sol:
        if s.denominator > 2:
            sol_ok = False
            break
    if not sol_ok:
        return None
    W_new = W_k * np.array([float(s) for s in sol]).reshape(1, -1)
    return U_k.copy(), V_k.copy(), W_new


def _try_drop_float(U, V, W, T, a, b, c, drop):
    """Float least-squares fallback for drop."""
    r = U.shape[1]
    keep = [d for d in range(r) if d != drop]
    U_k = U[:, keep]
    V_k = V[:, keep]
    W_k = W[:, keep]
    C_k = _coeff_matrix(U_k, V_k, W_k, a, b, c)
    T_flat = T.reshape(-1)
    try:
        sol, res, rankk, sv = np.linalg.lstsq(C_k, T_flat, rcond=1e-12)
    except Exception:
        return None
    residual = np.max(np.abs(C_k @ sol - T_flat))
    if residual > 1e-7:
        return None
    W_new = W_k * sol.reshape(1, -1)
    return U_k.copy(), V_k.copy(), W_new


def _reduce_by_drops(U, V, W, T, a, b, c, max_iter=30):
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    for _ in range(max_iter):
        if _time_left() < 5:
            break
        cur_r = best_U.shape[1]
        if cur_r <= 1:
            break
        found = False
        for drop in range(cur_r):
            if _time_left() < 2:
                break
            res = _try_drop_exact(best_U, best_V, best_W, T, a, b, c, drop)
            if res is None:
                res = _try_drop_float(best_U, best_V, best_W, T, a, b, c, drop)
            if res is not None:
                best_U, best_V, best_W = res
                found = True
                break
        if not found:
            break
    return best_U, best_V, best_W


def _try_pair_drop(U, V, W, T, a, b, c):
    r = U.shape[1]
    for c1 in range(r):
        if _time_left() < 3:
            break
        for c2 in range(c1 + 1, r):
            if _time_left() < 2:
                break
            keep = [d for d in range(r) if d != c1 and d != c2]
            U_k = U[:, keep]; V_k = V[:, keep]; W_k = W[:, keep]
            C_k = _coeff_matrix(U_k, V_k, W_k, a, b, c)
            T_flat = T.reshape(-1)
            sol = _exact_linear_solve(C_k, T_flat)
            if sol is None:
                continue
            sol_ok = all(s.denominator <= 2 for s in sol)
            if not sol_ok:
                continue
            W_new = W_k * np.array([float(s) for s in sol]).reshape(1, -1)
            return U_k.copy(), V_k.copy(), W_new
    return U, V, W


def _rebalance(U, V, W, T, a, b, c):
    """Try to find half-integer U,V,W with same rank by solving a balancing problem.
    For each column d, we can scale: U[:,d] *= s_d, V[:,d] *= t_d, W[:,d] *= u_d
    with s_d * t_d * u_d = 1. Try to find scalings that make all entries half-integers.
    Heuristic: for each column, choose scaling to minimize deviation from half-integers.
    """
    r = U.shape[1]
    Ur = U.copy(); Vr = V.copy(); Wr = W.copy()
    for d in range(r):
        # Current column entries
        uc = U[:, d]; vc = V[:, d]; wc = W[:, d]
        # Try scaling factor s on U, 1/s on W (keep V fixed)
        best_s = 1.0
        best_err = float('inf')
        for cand in [0.5, 1.0, 2.0, 0.25, 4.0, -0.5, -1.0, -2.0]:
            # New U col = uc * cand, W col = wc / cand
            nu = uc * cand
            nw = wc / cand
            err = np.sum(np.abs(nu * 2 - np.round(nu * 2))) + np.sum(np.abs(nw * 2 - np.round(nw * 2)))
            if err < best_err:
                best_err = err
                best_s = cand
        Ur[:, d] = uc * best_s
        Wr[:, d] = wc / best_s
    if _verify(Ur, Vr, Wr, T):
        return Ur, Vr, Wr
    return U, V, W


def _round_if_half(U, V, W, T):
    Ur = _round_half(U); Vr = _round_half(V); Wr = _round_half(W)
    if _verify(Ur, Vr, Wr, T):
        return Ur, Vr, Wr
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    a, b, c = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    U, V, W = _build_block_decomposition()

    if not _verify(U, V, W, T):
        r = n * m * p
        U_fb = np.zeros((n * m, r)); V_fb = np.zeros((m * p, r)); W_fb = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best = (U, V, W)

    # Iterative single-column drops (exact rational first, float fallback)
    if _time_left() > 5:
        U1, V1, W1 = _reduce_by_drops(U, V, W, T, a, b, c, max_iter=25)
        if U1.shape[1] < U.shape[1]:
            best = (U1, V1, W1)
            U, V, W = U1, V1, W1

    # Pair drops
    if _time_left() > 5:
        U2, V2, W2 = _try_pair_drop(U, V, W, T, a, b, c)
        if U2.shape[1] < U.shape[1]:
            best = (U2, V2, W2)
            U, V, W = U2, V2, W2
            # Continue single drops after pair
            if _time_left() > 5:
                U2b, V2b, W2b = _reduce_by_drops(U2, V2, W2, T, a, b, c, max_iter=15)
                if U2b.shape[1] < U2.shape[1]:
                    best = (U2b, V2b, W2b)
                    U, V, W = U2b, V2b, W2b

    # Rebalance to enable rounding
    if _time_left() > 3:
        Ur, Vr, Wr = _rebalance(U, V, W, T, a, b, c)
        if Ur.shape[1] == U.shape[1]:
            best = (Ur, Vr, Wr)
            U, V, W = Ur, Vr, Wr

    # Round to half-integers
    if _time_left() > 2:
        Ur, Vr, Wr = _round_if_half(U, V, W, T)
        if Ur.shape[1] == U.shape[1]:
            best = (Ur, Vr, Wr)
            U, V, W = Ur, Vr, Wr

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
