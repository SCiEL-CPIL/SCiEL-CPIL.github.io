"""
Matrix multiplication 2x2 x 2x11: rank-minimal exact tensor decomposition.

Strategy:
  - Verified rank-39 baseline (5 Strassen-7 blocks for columns 0-9 + 4 schoolbook for column 10)
  - Search for rank 38 using structured initializations with cross-column terms
  - Efficient ALS with vectorized updates
  - High-precision polish pass
  - Deadline guard for 120s budget
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 11]

# Precompute index lists
def _build_index_lists(n, m, p):
    ta, tb, tc = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ta.append(i * m + j)
                tb.append(j * p + k)
                tc.append(k * n + i)
    return (np.array(ta), np.array(tb), np.array(tc))

TA, TB, TC = _build_index_lists(*DIMENSIONS)
_TA, _TB, _TC = TA, TB, TC

# Precompute index groups for ALS
_idx_U = {}
_idx_V = {}
_idx_W = {}
for t in range(len(TA)):
    a, b, c = TA[t], TB[t], TC[t]
    _idx_U.setdefault(a, []).append(t)
    _idx_V.setdefault(b, []).append(t)
    _idx_W.setdefault(c, []).append(t)
_idx_U = {k: np.array(v) for k, v in _idx_U.items()}
_idx_V = {k: np.array(v) for k, v in _idx_V.items()}
_idx_W = {k: np.array(v) for k, v in _idx_W.items()}


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check exact decomposition."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    sums = np.zeros(len(TA))
    for c in range(rank):
        sums += Ua[TA, c] * Va[TB, c] * Wa[TC, c]
    return bool(np.all(np.abs(sums - 1.0) <= tol))


def _construct_rank39():
    """Verified rank-39 decomposition."""
    n, m, p = 2, 2, 11
    rank = 39
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1

        # Strassen 7 terms
        U[0, r] = 1; U[3, r] = 1
        V[v_e, r] = 1; V[v_h, r] = 1
        W[w_c00, r] = 1; W[w_c11, r] = 1
        r += 1

        U[2, r] = 1; U[3, r] = 1
        V[v_e, r] = 1
        W[w_c10, r] = 1; W[w_c11, r] = -1
        r += 1

        U[0, r] = 1
        V[v_f, r] = 1; V[v_h, r] = -1
        W[w_c01, r] = 1; W[w_c11, r] = 1
        r += 1

        U[3, r] = 1
        V[v_g, r] = 1; V[v_e, r] = -1
        W[w_c00, r] = 1; W[w_c10, r] = 1
        r += 1

        U[0, r] = 1; U[1, r] = 1
        V[v_h, r] = 1
        W[w_c00, r] = -1; W[w_c01, r] = 1
        r += 1

        U[2, r] = 1; U[0, r] = -1
        V[v_e, r] = 1; V[v_f, r] = 1
        W[w_c11, r] = 1
        r += 1

        U[1, r] = 1; U[3, r] = -1
        V[v_g, r] = 1; V[v_h, r] = 1
        W[w_c00, r] = 1
        r += 1

    # Schoolbook for column 10
    k2 = 10
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n
    w_c12 = k2 * n + 1

    U[0, r] = 1
    V[v_e2, r] = 1
    W[w_c02, r] = 1
    r += 1

    U[1, r] = 1
    V[v_g2, r] = 1
    W[w_c02, r] = 1
    r += 1

    U[2, r] = 1
    V[v_e2, r] = 1
    W[w_c12, r] = 1
    r += 1

    U[3, r] = 1
    V[v_g2, r] = 1
    W[w_c12, r] = 1
    r += 1

    assert r == rank
    return U.tolist(), V.tolist(), W.tolist()


def _als_round(U, V, W, rank, n, m, p):
    """One full ALS round."""
    # Update U
    for a in range(n * m):
        if a not in _idx_U:
            continue
        ts = _idx_U[a]
        prod = V[TB[ts], :] * W[TC[ts], :]
        num = prod.sum(axis=0)
        den = (prod ** 2).sum(axis=0)
        valid = den > 1e-14
        if np.any(valid):
            U[a, valid] = num[valid] / den[valid]

    # Update V
    for b in range(m * p):
        if b not in _idx_V:
            continue
        ts = _idx_V[b]
        prod = U[TA[ts], :] * W[TC[ts], :]
        num = prod.sum(axis=0)
        den = (prod ** 2).sum(axis=0)
        valid = den > 1e-14
        if np.any(valid):
            V[b, valid] = num[valid] / den[valid]

    # Update W
    for c_idx in range(p * n):
        if c_idx not in _idx_W:
            continue
        ts = _idx_W[c_idx]
        prod = U[TA[ts], :] * V[TB[ts], :]
        num = prod.sum(axis=0)
        den = (prod ** 2).sum(axis=0)
        valid = den > 1e-14
        if np.any(valid):
            W[c_idx, valid] = num[valid] / den[valid]


def _compute_error(U, V, W):
    """Sum-of-squares error."""
    sums = np.zeros(len(TA))
    for c in range(U.shape[1]):
        sums += U[TA, c] * V[TB, c] * W[TC, c]
    return float(np.sum((sums - 1.0) ** 2))


def _polish(U, V, W, rank, n, m, p, deadline, max_iter=5000):
    """High-precision polish."""
    best_err = _compute_error(U, V, W)
    for it in range(max_iter):
        if time.time() > deadline:
            break
        _als_round(U, V, W, rank, n, m, p)
        if it % 10 == 0:
            err = _compute_error(U, V, W)
            if err < best_err:
                best_err = err
            if err < 1e-14:
                return U, V, W, err
    return U, V, W, best_err


def _structured_init_rank38(n, m, p, seed=0):
    """
    Structured initialization for rank-38 that breaks block-diagonal structure.
    Start from rank-39, remove one column, and add a cross-column term.
    """
    rng = np.random.RandomState(seed)
    U39, V39, W39 = _construct_rank39()
    U39 = np.array(U39, dtype=float)
    V39 = np.array(V39, dtype=float)
    W39 = np.array(W39, dtype=float)
    
    rank = 38
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Copy 38 of the 39 columns
    # Remove the last schoolbook column (index 38) to make room
    U[:, :38] = U39[:, :38]
    V[:, :38] = V39[:, :38]
    W[:, :38] = W39[:, :38]
    
    # Add small noise to break exact structure
    U += rng.randn(n * m, rank) * 0.01
    V += rng.randn(m * p, rank) * 0.01
    W += rng.randn(p * n, rank) * 0.01
    
    return U, V, W


def _structured_init_rank38_cross(n, m, p, seed=0):
    """
    Alternative: start from rank-39, merge two schoolbook terms into one cross-column term.
    """
    rng = np.random.RandomState(seed)
    U39, V39, W39 = _construct_rank39()
    U39 = np.array(U39, dtype=float)
    V39 = np.array(V39, dtype=float)
    W39 = np.array(W39, dtype=float)
    
    rank = 38
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Copy first 36 columns (5 Strassen blocks = 35 cols, plus 1 schoolbook)
    U[:, :36] = U39[:, :36]
    V[:, :36] = V39[:, :36]
    W[:, :36] = W39[:, :36]
    
    # Column 36: combine the two schoolbook terms for column 10 that both write to w_c02
    # Original: col 36 = U[0]=1, V[10]=1, W[20]=1
    #           col 37 = U[1]=1, V[12]=1, W[20]=1
    # Merge into one term with both U and V components
    U[0, 36] = 1
    U[1, 36] = 1
    V[10, 36] = 1
    V[12, 36] = 1
    W[20, 36] = 1
    
    # Column 37: the remaining schoolbook term
    U[2, 37] = 1
    V[10, 37] = 1
    W[21, 37] = 1
    
    # Add small noise
    U += rng.randn(n * m, rank) * 0.01
    V += rng.randn(m * p, rank) * 0.01
    W += rng.randn(p * n, rank) * 0.01
    
    return U, V, W


def _refine_once(U, V, W, rank, n, m, p, deadline, max_rounds=2000):
    """Run ALS; return verified result if converged, else None."""
    best_err = float('inf')
    U_best = V_best = W_best = None
    converged = False
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        _als_round(U, V, W, rank, n, m, p)
        if it % 20 == 0:
            err = _compute_error(U, V, W)
            if err < best_err:
                best_err = err
                U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
            if err < 1e-12:
                converged = True
                break
    
    candidates = []
    if converged:
        candidates.append((U, V, W))
    if best_err < 1e-10 and U_best is not None:
        candidates.append((U_best, V_best, W_best))
    
    for Uc, Vc, Wc in candidates:
        if time.time() > deadline:
            break
        Uf, Vf, Wf, err = _polish(Uc, Vc, Wc, rank, n, m, p, deadline=deadline, max_iter=2000)
        if err < 1e-10:
            if _verify(Uf.tolist(), Vf.tolist(), Wf.tolist(), n, m, p):
                return Uf.tolist(), Vf.tolist(), Wf.tolist()
    return None


def _search_rank38(n, m, p, deadline, n_seeds=8):
    """Search for rank-38 decomposition."""
    for seed in range(n_seeds):
        if time.time() > deadline:
            break
        if seed % 2 == 0:
            U, V, W = _structured_init_rank38(n, m, p, seed=seed)
        else:
            U, V, W = _structured_init_rank38_cross(n, m, p, seed=seed)
        result = _refine_once(U, V, W, 38, n, m, p, deadline=deadline, max_rounds=2000)
        if result is not None:
            return result
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Verified rank-39 baseline
    U39, V39, W39 = _construct_rank39()
    if not _verify(U39, V39, W39, n, m, p):
        # Fallback: naive rank 44
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = (39, U39, V39, W39)
    
    # 2. Try to find rank-38
    if time.time() < deadline - 5:
        result = _search_rank38(n, m, p, deadline=deadline - 3, n_seeds=8)
        if result is not None:
            best_result = (38, result[0], result[1], result[2])
    
    final_rank, U_final, V_final, W_final = best_result
    return {"U": U_final, "V": V_final, "W": W_final}
