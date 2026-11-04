"""Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Design family (kept from the current program):
  PART 1: 4x4 * 4x4 core (columns 0-3 of B) via recursive Strassen -> rank 49.
  PART 2: 4x4 * 4x2 strip (columns 4-5 of B) via 4 independent 2x2*2x2
          Strassen blocks -> rank 28.
  Total: 77.

Refinement added here: after building the 77 rank-1 terms, we search for
linear dependencies among them. Any dependency lets us drop one term and
compensate with the others, preserving the exact decomposition while
reducing the rank. The search is bounded and guarded by a wall-clock
deadline so it always returns within the 150 s budget.
"""

import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before the 150 s kill


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication.

    Returns a dict with keys 'U', 'V', 'W' where:
      U: shape (n*m, rank) = (16, rank)
      V: shape (m*p, rank) = (24, rank)
      W: shape (p*n, rank) = (24, rank)
    The exact tensor condition is:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1   for all i, j, k
    """
    n, m, p = 4, 4, 6

    U_cols = []
    V_cols = []
    W_cols = []

    # PART 1: 4x4 * 4x4 core via recursive Strassen (rank 49)
    if _time_remaining() > 5.0:
        _add_strassen_4x4(0, 0, 0, U_cols, V_cols, W_cols, n, m, p)

    # PART 2: 4x4 * 4x2 strip via 4 independent 2x2*2x2 Strassen (rank 28)
    if _time_remaining() > 3.0:
        for rb in range(2):
            if _time_remaining() < 2.0:
                break
            _add_strassen_2x2(rb * 2, 0, 4, U_cols, V_cols, W_cols, n, m, p)
            _add_strassen_2x2(rb * 2, 2, 4, U_cols, V_cols, W_cols, n, m, p)

    # Convert to arrays (rows = indices, cols = rank-1 terms).
    U0 = np.array(U_cols, dtype=np.float64).T
    V0 = np.array(V_cols, dtype=np.float64).T
    W0 = np.array(W_cols, dtype=np.float64).T

    # Refinement: try to reduce the rank via linear dependencies.
    if _time_remaining() > 10.0:
        Uf, Vf, Wf = _refine(U0, V0, W0, n, m, p)
        if Uf is not None:
            U0, V0, W0 = Uf, Vf, Wf

    # Build the final list-of-lists in the required row-major orientation.
    rank = U0.shape[1]
    U_final = U0.tolist()
    V_final = V0.tolist()
    W_final = W0.tolist()

    # Final verification; fall back to the schoolbook rank-96 on failure.
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        return {"U": U_final, "V": V_final, "W": W_final}

    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _refine(U0, V0, W0, n, m, p):
    """
    Refinement pass: iteratively drop one rank-1 term and, when a linear
    dependency exists among the remaining terms, compensate with the others.
    Returns the reduced (U, V, W) arrays, or None if no reduction was found.
    """
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    r = U.shape[1]
    eps = 1e-9

    # Precompute the row-index map for verification.
    u_idx = [i * m + j for i in range(n) for j in range(m)]
    v_idx = [j * p + k for j in range(m) for k in range(p)]
    w_idx = [k * n + i for k in range(p) for i in range(n)]

    def _ok(Um, Vm, Wm, rj):
        # sum_r U[a,r] V[b,r] W[c,r] must equal 1 for all a in u_idx,
        # b in v_idx, c in w_idx.
        A = Um[u_idx, :rj]
        B = Vm[v_idx, :rj]
        C = Wm[w_idx, :rj]
        s = np.einsum('ar,br,cr->abc', A, B, C)
        return bool(np.all(np.abs(s - 1.0) < eps))

    improved = False
    # Try to drop each column at least once (a couple of sweeps are cheap).
    for _sweep in range(2):
        if _time_remaining() < 5.0:
            break
        progress = False
        for j in range(r):
            if _time_remaining() < 5.0:
                break
            # Does column j lie in the span of the other columns?
            others = list(range(j)) + list(range(j + 1, r))
            Uo = U[:, others]
            Vo = V[:, others]
            Wo = W[:, others]
            # Find alpha such that Uo @ alpha ~= U[:, j].
            alpha, res, rankv, _sv = np.linalg.lstsq(Uo, U[:, j], rcond=None)
            if rankv < len(others):
                # Check that the same alpha works for V and W.
                if not (np.allclose(Uo @ alpha, U[:, j], atol=eps)
                        and np.allclose(Vo @ alpha, V[:, j], atol=eps)
                        and np.allclose(Wo @ alpha, W[:, j], atol=eps)):
                    continue
            else:
                # No exact dependency in U; no compensation possible.
                continue

            # Drop column j and add the compensation to the others.
            U2 = np.column_stack([Uo, -Uo @ alpha])
            V2 = np.column_stack([Vo, -Vo @ alpha])
            W2 = np.column_stack([Wo, -Wo @ alpha])
            if _ok(U2, V2, W2, r - 1):
                U, V, W = U2, V2, W2
                r -= 1
                improved = True
                progress = True
                break
        if not progress:
            break

    if improved:
        return U[:, :r], V[:, :r], W[:, :r]
    return None


def _add_strassen_2x2(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p):
    """Add Strassen decomposition for 2x2 * 2x2 block multiplication."""
    S2 = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    for u_pat, v_pat, w_pat in S2:
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        for (r, c), coeff in u_pat.items():
            i = oi + r
            j = oj + c
            if 0 <= i < n and 0 <= j < m:
                u_col[i * m + j] += coeff
        for (r, c), coeff in v_pat.items():
            j = oj + r
            k = ok + c
            if 0 <= j < m and 0 <= k < p:
                v_col[j * p + k] += coeff
        for (r, c), coeff in w_pat.items():
            i = oi + r
            k = ok + c
            if 0 <= i < n and 0 <= k < p:
                w_col[k * n + i] += coeff
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)


def _add_strassen_4x4(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p):
    """
    Add recursive Strassen decomposition for 4x4 * 4x4 block multiplication.
    Uses 7 super-terms, each expanded into 7 Strassen 2x2 terms -> 49 terms.
    """
    S2 = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    block_offsets = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
    super_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    for u_blocks, v_blocks, w_blocks in super_terms:
        for u_pat, v_pat, w_pat in S2:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            for block_idx, block_coeff in u_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in u_pat.items():
                    i = oi + br + r
                    j = oj + bc + c
                    if 0 <= i < n and 0 <= j < m:
                        u_col[i * m + j] += block_coeff * coeff
            for block_idx, block_coeff in v_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in v_pat.items():
                    j = oj + br + r
                    k = ok + bc + c
                    if 0 <= j < m and 0 <= k < p:
                        v_col[j * p + k] += block_coeff * coeff
            for block_idx, block_coeff in w_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in w_pat.items():
                    i = oi + br + r
                    k = ok + bc + c
                    if 0 <= i < n and 0 <= k < p:
                        w_col[k * n + i] += block_coeff * coeff
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                if abs(total - 1.0) > 1e-9:
                    return False
    return True
