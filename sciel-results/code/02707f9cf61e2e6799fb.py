"""Rank-minimal exact tensor decomposition for 3x3 x 3x4 matrix multiplication.

Searches for integer/half-integer factor matrices U, V, W that exactly
decompose the multiplication tensor, minimizing rank.
"""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 4]
_T0 = time.time()
_LIMIT = 135.0  # safety margin below 150s kill


def _time_left():
    return _LIMIT - (time.time() - _T0)


def _build_tensor(n, m, p):
    """Return tensor T[i,k,j] = 1 iff (i,k) in A, (k,j) in B, (i,j) in C."""
    T = np.zeros((n * n, m * m, p * p), dtype=np.int64)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T[i * n + j, k * m + i, k * m + j] = 1
    return T


def _verify(U, V, W, n, m, p):
    """Check exact decomposition. Returns (ok, error)."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    R = Ua.shape[1]
    if Va.shape[1] != R or Wa.shape[1] != R:
        return False, "shape mismatch"
    if Ua.shape[0] != n * m or Va.shape[0] != m * p or Wa.shape[0] != p * n:
        return False, "shape mismatch"
    T = np.zeros((n * n, m * m, p * p), dtype=np.float64)
    for r in range(R):
        u = Ua[:, r]
        v = Va[:, r]
        w = Wa[:, r]
        T += np.einsum('a,b,c->abc', u, v, w)
    # expected tensor
    Et = _build_tensor(n, m, p).astype(np.float64)
    err = np.max(np.abs(T - Et))
    return err < 1e-6, err


def _is_halfint(a):
    a = np.array(a, dtype=np.float64)
    return np.allclose(a, np.round(2 * a) / 2.0, atol=1e-9)


def _score_of(rank):
    # seed program rank = 34
    return 34.0 / rank


def _search_small_rank(n, m, p, max_rank):
    """Brute-force search for integer {0,1} decompositions at low rank."""
    T = _build_tensor(n, m, p)
    # candidate nonzero patterns for U (n*m rows), V (m*p rows), W (p*n rows)
    # For rank r, each column is a vector. We'll enumerate a limited set.
    # This is only practical for very small r; use as a sanity probe.
    nA = n * m
    nB = m * p
    nC = p * n
    # We'll try to find rank <= max_rank using a randomized/structured approach.
    # (Full brute force is infeasible; this is a probe.)
    return None


def construct():
    n, m, p = DIMENSIONS  # 3,3,4
    nA, nB, nC = n * m, m * p, p * n  # 9, 12, 12
    T = _build_tensor(n, m, p)

    best = None
    best_rank = None

    # --- Try to find a rank-33 decomposition via structured search ---
    # The rank-34 solution uses two Strassen blocks (7 each) + 20 direct terms.
    # We attempt to merge one direct term into a Strassen product by allowing
    # a half-integer coefficient, or to eliminate a term via linearity.

    # Strategy: start from the rank-34 matrix, then attempt to nullify one
    # column by adding a linear combination of the others (integer kernel).
    # If a column is in the span of the others over Q, we can drop it.

    def _build_34():
        rank = 34
        U = [[0] * rank for _ in range(nA)]
        V = [[0] * rank for _ in range(nB)]
        W = [[0] * rank for _ in range(nC)]
        col = 0

        def add(uv, vv, wv):
            nonlocal col
            for idx, val in uv:
                U[idx][col] += val
            for idx, val in vv:
                V[idx][col] += val
            for idx, val in wv:
                W[idx][col] += val
            col += 1

        # Strassen Left (A11 * B1L)
        add([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (4, 1)])
        add([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (4, -1)])
        add([(0, 1)], [(1, 1), (5, -1)], [(3, 1), (4, 1)])
        add([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)])
        add([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (3, 1)])
        add([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(4, 1)])
        add([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)])

        # A12 * B2L
        add([(2, 1)], [(8, 1)], [(0, 1)])
        add([(2, 1)], [(9, 1)], [(3, 1)])
        add([(5, 1)], [(8, 1)], [(1, 1)])
        add([(5, 1)], [(9, 1)], [(4, 1)])

        # A21 * B1L
        add([(6, 1)], [(0, 1)], [(2, 1)])
        add([(6, 1)], [(1, 1)], [(5, 1)])
        add([(7, 1)], [(4, 1)], [(2, 1)])
        add([(7, 1)], [(5, 1)], [(5, 1)])

        # A22 * B2L
        add([(8, 1)], [(8, 1)], [(2, 1)])
        add([(8, 1)], [(9, 1)], [(5, 1)])

        # Strassen Right (A11 * B1R)
        add([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(6, 1), (10, 1)])
        add([(3, 1), (4, 1)], [(2, 1)], [(7, 1), (10, -1)])
        add([(0, 1)], [(3, 1), (7, -1)], [(9, 1), (10, 1)])
        add([(4, 1)], [(6, 1), (2, -1)], [(6, 1), (7, 1)])
        add([(0, 1), (1, 1)], [(7, 1)], [(6, -1), (9, 1)])
        add([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(10, 1)])
        add([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(6, 1)])

        # A12 * B2R
        add([(2, 1)], [(10, 1)], [(6, 1)])
        add([(2, 1)], [(11, 1)], [(9, 1)])
        add([(5, 1)], [(10, 1)], [(7, 1)])
        add([(5, 1)], [(11, 1)], [(10, 1)])

        # A21 * B1R
        add([(6, 1)], [(2, 1)], [(8, 1)])
        add([(6, 1)], [(3, 1)], [(11, 1)])
        add([(7, 1)], [(6, 1)], [(8, 1)])
        add([(7, 1)], [(7, 1)], [(11, 1)])

        # A22 * B2R
        add([(8, 1)], [(10, 1)], [(8, 1)])
        add([(8, 1)], [(11, 1)], [(11, 1)])

        return U, V, W

    U34, V34, W34 = _build_34()
    ok, err = _verify(U34, V34, W34, n, m, p)
    if ok:
        best = (U34, V34, W34)
        best_rank = 34

    # --- Attempt to reduce rank by finding a rational linear dependency ---
    # Stack columns: for each column r, the "signature" is the triple
    # (U[:,r], V[:,r], W[:,r]) as a flat vector of length nA+nB+nC.
    # If columns are linearly dependent over Q, we can eliminate one.
    # But the decomposition requires the *tensor* to match, not just columns.
    # A column c = sum_k alpha_k c_k gives tensor contribution
    # sum_k alpha_k T_k. This equals zero iff the tensor contributions cancel.
    # We check: is there a set of coefficients alpha (not all zero) such that
    # sum_k alpha_k * (U_k, V_k, W_k) as a tensor = 0?
    # Equivalently, the tensors T_k = outer(U_k, V_k, W_k) are linearly dependent.
    # We check rank of the tensor matrix.

    R = best_rank
    # Build tensor for each column and check linear dependence
    flat = []
    for r in range(R):
        u = np.array(U34[:, r], dtype=np.float64)
        v = np.array(V34[:, r], dtype=np.float64)
        w = np.array(W34[:, r], dtype=np.float64)
        t = np.einsum('a,b,c->abc', u, v, w)
        flat.append(t.flatten())
    M = np.array(flat).T  # (nA*nB*nC, R)
    # Check if columns are dependent
    try:
        rank_M = np.linalg.matrix_rank(M, tol=1e-9)
    except Exception:
        rank_M = R
    if rank_M < R:
        # There's a dependency; try to eliminate a column
        # Find null space vector
        try:
            _, S, Vt = np.linalg.svd(M)
            null_dim = R - rank_M
            # null space is last null_dim rows of Vt
            for ns in range(null_dim):
                alpha = Vt[R - 1 - ns, :]
                # find a column with large |alpha| to drop
                idx = int(np.argmax(np.abs(alpha)))
                # new decomposition: drop column idx, scale others
                newR = R - 1
                Unew = [row[:] for row in U34]
                Vnew = [row[:] for row in V34]
                Wnew = [row[:] for row in W34]
                # remove column idx: shift columns
                newU = [[0] * newR for _ in range(nA)]
                newV = [[0] * newR for _ in range(nB)]
                newW = [[0] * newR for _ in range(nC)]
                src = 0
                for r in range(R):
                    if r == idx:
                        continue
                    for a in range(nA):
                        newU[a][src] = U34[a][r]
                    for b in range(nB):
                        newV[b][src] = V34[b][r]
                    for c in range(nC):
                        newW[c][src] = W34[c][r]
                    src += 1
                ok2, err2 = _verify(newU, newV, newW, n, m, p)
                if ok2 and _is_halfint(newU) and _is_halfint(newV) and _is_halfint(newW):
                    best = (newU, newV, newW)
                    best_rank = newR
                    break
        except Exception:
            pass

    # --- Structured search: try to find a rank-33 via targeted modification ---
    # Attempt: combine two direct terms into one by sharing a factor.
    # This is heuristic; we try a small set of candidate merges.
    if _time_left() > 30:
        # Try a randomized search for a lower-rank decomposition.
        # We'll use a simple local search: start from rank-34, try to
        # replace a column with a half-integer linear combination.
        rng = np.random.RandomState(42)
        for attempt in range(200):
            if _time_left() < 30:
                break
            # Pick a column to try to eliminate
            drop = rng.randint(R)
            # Try to express column drop as combination of others with half-integers
            # Solve M_without_drop * x = M_drop
            cols = [c for c in range(R) if c != drop]
            Msub = M[:, cols]
            b = M[:, drop]
            try:
                x, residuals, rank_x, sv = np.linalg.lstsq(Msub, b, rcond=1e-9)
            except Exception:
                continue
            # Check if x is close to half-integers and residual small
            if np.max(np.abs(Msub @ x - b)) < 1e-6:
                # Check x is half-integer
                x2 = np.round(2 * x) / 2.0
                if np.max(np.abs(x - x2)) < 1e-6:
                    # Build new decomposition
                    newR = R - 1
                    newU = [[0] * newR for _ in range(nA)]
                    newV = [[0] * newR for _ in range(nB)]
                    newW = [[0] * newR for _ in range(nC)]
                    src = 0
                    for r in cols:
                        for a in range(nA):
                            newU[a][src] = U34[a][r]
                        for b_idx in range(nB):
                            newV[b_idx][src] = V34[b_idx][r]
                        for c_idx in range(nC):
                            newW[c_idx][src] = W34[c_idx][r]
                        src += 1
                    ok3, err3 = _verify(newU, newV, newW, n, m, p)
                    if ok3 and _is_halfint(newU) and _is_halfint(newV) and _is_halfint(newW):
                        best = (newU, newV, newW)
                        best_rank = newR
                        break

    if best is None:
        # Fallback to rank-34
        best = (U34, V34, W34)

    U, V, W = best
    return {"U": U, "V": V, "W": W}
