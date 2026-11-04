"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    W uses trace convention: W[k*n+i] with n=2.
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)

    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1

    # W: C[i,k], row = k*2 + i
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[2, 2] = 1; W[2, 4] = 1
    W[1, 1] = 1; W[1, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    return U, V, W


def _build_base_rank77():
    """
    Build the rank-77 decomposition: 11 outer terms x 7 inner Strassen.
    Outer: 7 Strassen terms for cols 0,1 + 4 schoolbook terms for col 2.
    """
    sU, sV, sW = _strassen_2x2()
    n, m, p = DIMENSIONS  # 4, 4, 6

    outer_terms = []
    # 7 Strassen terms for cols 0,1
    for t in range(7):
        U_dict = {}
        V_dict = {}
        W_dict = {}
        for a in range(2):
            for b in range(2):
                coeff = sU[a * 2 + b, t]
                if coeff != 0:
                    U_dict[(a, b)] = U_dict.get((a, b), 0) + coeff
        for b in range(2):
            for c in range(2):
                coeff = sV[b * 2 + c, t]
                if coeff != 0:
                    V_dict[(b, c)] = V_dict.get((b, c), 0) + coeff
        for a in range(2):
            for c in range(2):
                coeff = sW[c * 2 + a, t]
                if coeff != 0:
                    W_dict[(a, c)] = W_dict.get((a, c), 0) + coeff
        outer_terms.append((U_dict, V_dict, W_dict))
    # 4 schoolbook terms for col 2
    for a in range(2):
        for b in range(2):
            outer_terms.append({(a, b): 1}, {(b, 2): 1}, {(a, 2): 1})

    total_rank = len(outer_terms) * 7
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    for ot_idx, (U_dict, V_dict, W_dict) in enumerate(outer_terms):
        for t in range(7):
            col = ot_idx * 7 + t
            for (ai, bj), coeff_outer in U_dict.items():
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_inner = sU[ai2 * 2 + bj2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            u_row = (2 * ai + ai2) * 4 + (2 * bj + bj2)
                            U[u_row, col] += coeff_outer * coeff_inner
            for (bj, ck), coeff_outer in V_dict.items():
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sV[bj2 * 2 + ck2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            v_row = (2 * bj + bj2) * 6 + (2 * ck + ck2)
                            V[v_row, col] += coeff_outer * coeff_inner
            for (ai, ck), coeff_outer in W_dict.items():
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2 * 2 + ai2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            w_row = (2 * ck + ck2) * 4 + (2 * ai + ai2)
                            W[w_row, col] += coeff_outer * coeff_inner
    return U, V, W


def _build_tensors(n, m, p):
    """Build index arrays and the target multiplication tensor (flattened)."""
    idx = np.arange(n * m * p)
    i_idx = (idx // m) % n
    j_idx = (idx // p) % m
    k_idx = idx % p
    target = np.ones(n * m * p, dtype=float)
    return i_idx, j_idx, k_idx, target


def _try_remove_column(U, V, W, i_idx, j_idx, k_idx, target, col):
    """
    Try to remove column `col` by re-absorbing its contribution into the other
    columns. Returns (success, newU, newV, newW).
    """
    R = U.shape[1]
    keep = [r for r in range(R) if r != col]
    Uk = U[:, keep]
    Vk = V[:, keep]
    Wk = W[:, keep]

    u_col = U[:, col]
    v_col = V[:, col]
    w_col = W[:, col]

    # Contribution of column col to the tensor
    contrib = (u_col[i_idx] * v_col[j_idx]) * w_col[k_idx]
    residual = target - contrib

    # Build linear system: residual = sum_{r in keep} (Uk[:,r] * Vk[:,r]) * Wk[:,r] * delta_W[r]
    # i.e. X @ delta_W = residual, where X[s, r] = Uk[i_idx[s], r] * Vk[j_idx[s], r] * Wk[k_idx[s], r]
    X = (Uk[i_idx, :] * Vk[j_idx, :]) * Wk[k_idx, :]
    try:
        delta_W, *_ = np.linalg.lstsq(X, residual, rcond=None)
    except Exception:
        return False, None, None, None

    # New W columns
    W_new = Wk.copy()
    W_new[:, :] += Wk * delta_W  # Wk[r] * (1 + delta_W[r])

    # Verify exactly
    recon = np.einsum('sr,rr2,sr2->s', Uk, Vk, W_new)
    if not np.allclose(recon, target, atol=1e-9, rtol=1e-9):
        return False, None, None, None

    return True, Uk, Vk, W_new


def _rank_reduce(U, V, W, deadline):
    """Greedy coordinate descent: repeatedly remove the removable column."""
    n, m, p = DIMENSIONS
    i_idx, j_idx, k_idx, target = _build_tensors(n, m, p)

    # Precompute per-column "sparsity" heuristic: prefer removing columns with
    # fewest non-zero entries (cheaper to re-absorb / more likely redundant).
    def col_weight(U, V, W, r):
        return (np.count_nonzero(U[:, r]) + np.count_nonzero(V[:, r]) + np.count_nonzero(W[:, r]))

    while U.shape[1] > 1:
        if time.time() > deadline:
            break
        R = U.shape[1]
        # pick the candidate column with smallest weight
        weights = [col_weight(U, V, W, r) for r in range(R)]
        order = sorted(range(R), key=lambda r: weights[r])
        removed = False
        for col in order:
            ok, nU, nV, nW = _try_remove_column(U, V, W, i_idx, j_idx, k_idx, target, col)
            if ok:
                U, V, W = nU, nV, nW
                removed = True
                break
        if not removed:
            break
    return U, V, W


def _to_int(mat):
    """Convert a float matrix that should be integer/half-integer to a clean int list."""
    a = np.asarray(mat, dtype=float)
    return np.rint(a).astype(int).tolist()


def _verify_full(U, V, W):
    n, m, p = DIMENSIONS
    i_idx, j_idx, k_idx, target = _build_tensors(n, m, p)
    recon = np.einsum('sr,rr2,sr2->s', U, V, W)
    return bool(np.allclose(recon, target, atol=1e-9, rtol=1e-9))


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    U, V, W = _build_base_rank77()

    # Sanity check the base
    if not _verify_full(U, V, W):
        # base should always be valid; if not, return it anyway
        return {"U": _to_int(U), "V": _to_int(V), "W": _to_int(W)}

    # Rank reduction via coordinate descent
    U, V, W = _rank_reduce(U, V, W, deadline)

    # Final exactness check; if reduction broke it (shouldn't), fall back to base
    if not _verify_full(U, V, W):
        U, V, W = _build_base_rank77()

    return {"U": _to_int(U), "V": _to_int(V), "W": _to_int(W)}

# EVOLVE-BLOCK-END
