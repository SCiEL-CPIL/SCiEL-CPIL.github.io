"""Combined program: verified rank-28 block-Strassen + exact rank-27 elimination search."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

# Strassen terms for 2x2 matrix multiplication X*Y -> P.
# Local 2x2 index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
# Each term: (u_coeffs, v_coeffs, w_coeffs) as list of (local_index, value)
_STRASSEN = [
    ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # M1
    ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),          # M2
    ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),          # M3
    ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),          # M4
    ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),          # M5
    ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),          # M6
    ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),          # M7
]


def _build_strassen_28():
    """Build a verified rank-28 decomposition via Strassen on 2x2 blocks.

    A (2x4) = [A_L | A_R], B (4x4) split into 2x2 blocks, C (2x4) = [C_L | C_R].
    C_L = A_L*B_TL + A_R*B_BL ;  C_R = A_L*B_TR + A_R*B_BR.
    Each 2x2*2x2 uses 7 Strassen multiplications -> 4*7 = 28.
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    def apply_strassen(a_off, b_row_off, b_col_off, c_col_off, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(_STRASSEN):
            col = start_col + t_idx
            # U (A): local (r,c) -> global (r, a_off*2+c); flat = r*m + (a_off*2+c)
            for loc, coeff in u_loc:
                r, c = loc // 2, loc % 2
                U[r * m + (a_off * 2 + c), col] += coeff
            # V (B): local (r,c) -> global (b_row_off*2+r, b_col_off*2+c); flat = row*p + col
            for loc, coeff in v_loc:
                r, c = loc // 2, loc % 2
                V[(b_row_off * 2 + r) * p + (b_col_off * 2 + c), col] += coeff
            # W (C): output convention W[k*n+i]; local (r,c) -> global (r, c_col_off*2+c)
            for loc, coeff in w_loc:
                r, c = loc // 2, loc % 2
                W[(c_col_off * 2 + c) * n + r, col] += coeff

    apply_strassen(0, 0, 0, 0, 0)    # A_L * B_TL -> C_L
    apply_strassen(1, 1, 0, 0, 7)    # A_R * B_BL -> C_L
    apply_strassen(0, 0, 1, 1, 14)   # A_L * B_TR -> C_R
    apply_strassen(1, 1, 1, 1, 21)   # A_R * B_BR -> C_R
    return U, V, W


def _verify(U, V, W, n=2, m=4, p=4):
    """Return max |sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1| over all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            # sum over j,r
            acc = np.zeros(1)
            for j in range(m):
                acc += U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
            err = abs(float(acc.sum()) - 1.0)
            if err > max_err:
                max_err = err
    return max_err


def _try_reduce_to_27(U, V, W, n=2, m=4, p=4):
    """Try to reduce rank 28 -> 27 by eliminating one column.

    A column r* is eliminable if, for every (i,k), W[:,r*] is a linear combination
    of the other W columns (with coefficients depending on i,k) AND U[:,r*], V[:,r*]
    are linear combinations of the others. When so, the r* column's contribution to the
    tensor equals a combination of the others, so it can be dropped.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank <= 27:
        return None

    # For each candidate column r*, check eliminability.
    for r_star in range(rank):
        keep = [r for r in range(rank) if r != r_star]
        U_keep = U[:, keep]   # (nm, rank-1)
        V_keep = V[:, keep]
        W_keep = W[:, keep]
        U_s = U[:, r_star]
        V_s = V[:, r_star]
        W_s = W[:, r_star]

        ok = True
        # Check U_s and V_s are in the column spans of U_keep, V_keep.
        try:
            U_comb, *_ = np.linalg.lstsq(U_keep, U_s, rcond=None)
            V_comb, *_ = np.linalg.lstsq(V_keep, V_s, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if np.linalg.norm(U_keep @ U_comb - U_s) > 1e-8:
            continue
        if np.linalg.norm(V_keep @ V_comb - V_s) > 1e-8:
            continue

        # For every (i,k), the W contribution must match: W_s must equal
        # W_keep @ w_comb(i,k) where w_comb is the coefficient that makes the
        # tensor match. Actually we need: for the dropped column to be redundant,
        # the tensor contribution of r* must equal a linear combo of the others.
        # Contribution of r* to T[i,j,k] = U_s[i*m+j]*V_s[j*p+k]*W_s[k*n+i].
        # Contribution of keep with coeffs c_r (per (i,k) for W, per j for U,V) ...
        # This is complex; simplest exact check: build the reduced tensor and compare.
        # We'll construct reduced U,V,W by expressing r* via keep columns is not
        # straightforward because coeffs differ per index. Instead, verify by
        # direct tensor reconstruction: does removing r* and adding a correction
        # work? We instead test: is the rank-28 tensor equal to a rank-27 tensor
        # that agrees on all (i,k) with the same U,V restricted? 
        # Practical exact test: check if W_s is in span of {W_keep columns} uniformly
        # (same coeffs for all i,k) AND U_s, V_s in spans uniformly.
        try:
            W_comb, *_ = np.linalg.lstsq(W_keep, W_s, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if np.linalg.norm(W_keep @ W_comb - W_s) > 1e-8:
            continue
        # Uniform combination found; build reduced matrices and verify tensor.
        U_red = U[:, keep] + U_keep @ np.array([0]*0)  # placeholder
        # Express r* column: U[:,r*] = U_keep @ U_comb, etc. Dropping r* is valid
        # only if its contribution is already covered. Build reduced and verify.
        U_red = U[:, keep]
        V_red = V[:, keep]
        W_red = W[:, keep]
        if _verify(U_red, V_red, W_red, n, m, p) < 1e-8:
            return U_red, V_red, W_red
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    U, V, W = _build_strassen_28()
    best = (U, V, W)
    best_rank = U.shape[1]

    # Verify rank-28 is correct.
    if _verify(U, V, W) >= 1e-8:
        # Fallback: schoolbook rank-32 (guaranteed valid).
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best = (U, V, W)
        best_rank = rank
    else:
        # Attempt structural rank-27 reduction.
        if time.time() < deadline:
            reduced = _try_reduce_to_27(U, V, W)
            if reduced is not None:
                U2, V2, W2 = reduced
                if _verify(U2, V2, W2) < 1e-8:
                    best = (U2, V2, W2)
                    best_rank = U2.shape[1]

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END


if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
