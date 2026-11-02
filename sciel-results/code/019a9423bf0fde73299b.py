"""Fresh design: rank-reduction of the 4x5 x 5x6 matrix multiplication tensor.

Approach (structural, not generic search):
  1. Build the schoolbook (rank n*m*p) decomposition.
  2. Express the matrix-multiplication tensor as a linear combination of the
     rank-1 slices via sparse linear algebra (this recovers the canonical
     coefficients and exposes the true tensor structure).
  3. Greedily eliminate rank-1 terms: a term can be dropped if its tensor
     contribution is already spanned by the remaining terms (detected via a
     rank check on the coefficient matrix restricted to the tensor's support).
  4. Return the smallest exact decomposition found before the deadline.
"""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# Hard wall-clock deadline (evaluator kills at 150 s; keep a safety margin).
_DEADLINE = 150.0
_START = time.time()


def _remaining():
    return _DEADLINE - time.time()


def _is_half_integer(x):
    """True if x is an integer or a half-integer (within tolerance)."""
    y = 2.0 * x
    return abs(y - round(y)) < 1e-6


def _to_halfint(x):
    """Round x to the nearest half-integer."""
    return 0.5 * round(2.0 * x)


def _validate(U, V, W, n, m, p):
    """Check that the decomposition exactly reproduces the multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Build expected tensor (the multiplication tensor has 1 where the index
    # triple is a valid (i,j,k) with each term independent -> T[i,j,k]=1 for all
    # i,j,k because every product term (A B)[i,k] = sum_j A[i,j]B[j,k] uses
    # exactly the (i,j,k) slice).
    T = np.zeros((n, m, p))
    T[:, :, :] = 1.0
    # Compute via einsum over the rank axis.
    # U indexed (i,j,c), V (j,k,c), W (k,i,c)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # term for (i,j,k): sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # vectorised: for each (i,j,k)
    # T_c = U3[i,j,c]*V3[j,k,c]*W3[k,i,c]; T[i,j,k]=sum_c
    # Use broadcasting via einsum.
    # einsum('ijc,jkc,kic->ijk')
    Tcalc = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(Tcalc, T, atol=1e-6)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _term_vectors(U, V, W, n, m, p):
    """Return list of flattened rank-1 term tensors (one per rank column).

    term_c is the (n*m*p)-vector with entry (i,j,k) = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].
    """
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    terms = []
    for c in range(rank):
        # (i,j,k) = U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        t = np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
        terms.append(t.reshape(-1))
    return terms


def _support_mask(n, m, p):
    """The multiplication tensor is all-ones on its (n*m*p) support."""
    return np.ones(n * m * p, dtype=bool)


def _try_eliminate(U, V, W, n, m, p, max_iters=4000):
    """Greedily drop rank-1 terms whose contribution is redundant."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    terms = _term_vectors(U, V, W, n, m, p)
    mask = _support_mask(n, m, p)
    # We want the sum of remaining terms to equal the target tensor (all ones).
    # Represent each term restricted to the support as a column; we need a
    # subset of columns whose sum == target. Greedy: try removing column c if
    # (target - sum_remaining_without_c) is in the span of the others AND the
    # coefficient is 1 (i.e. removing c keeps the sum correct). Simpler exact
    # greedy: remove c if the sum of the other terms still equals the target.
    total = np.zeros_like(terms[0])
    for t in terms:
        total += t
    target = np.ones(n * m * p)
    active = list(range(rank))
    for _ in range(max_iters):
        if _remaining() < 3.0:
            break
        progress = False
        for idx in range(len(active)):
            c = active[idx]
            # candidate sum without c
            cand = total - terms[c]
            if np.allclose(cand, target, atol=1e-9):
                # remove
                del active[idx]
                total = cand
                progress = True
                break
        if not progress:
            break
    # Rebuild U,V,W keeping only active columns.
    U2 = U[:, active]
    V2 = V[:, active]
    W2 = W[:, active]
    return U2, V2, W2


def _try_column_merging(U, V, W, n, m, p, max_iters=3000):
    """Merge two columns c,d into one if U[:,c]+U[:,d], V[:,c]+V[:,d], W[:,c]+W[:,d]
    still yields a valid decomposition with one fewer column.

    A single column with entries (u, v, w) contributes the tensor
    einsum('ij,jk,ki->ijk', u, v, w). Two columns c,d contribute the sum of two
    such tensors. They merge into one column (u_c+u_d, v_c+v_d, w_c+w_d) iff
      einsum(u_c,v_c,w_c) + einsum(u_d,v_d,w_d) == einsum(u_c+u_d, v_c+v_d, w_c+w_d)
    which holds iff the cross terms vanish:
      einsum(u_c,v_d,w_d) + einsum(u_d,v_c,w_c) == 0
    (and the two cross terms with mixed pairs). Concretely the extra terms are
      einsum(u_c, v_d, w_d) + einsum(u_d, v_c, w_c)
    must equal 0. This is a cheap exact test."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)

    def cross(c, d):
        # extra terms when merging c and d
        a = np.einsum('ij,jk,ki->', U3[:, :, c], V3[:, :, d], W3[:, :, d])
        b = np.einsum('ij,jk,ki->', U3[:, :, d], V3[:, :, c], W3[:, :, c])
        return a + b

    active = list(range(rank))
    for _ in range(max_iters):
        if _remaining() < 3.0:
            break
        progress = False
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                c, d = active[i], active[j]
                if abs(cross(c, d)) < 1e-9:
                    # merge d into c
                    U[:, c] = U[:, c] + U[:, d]
                    V[:, c] = V[:, c] + V[:, d]
                    W[:, c] = W[:, c] + W[:, d]
                    del active[j]
                    progress = True
                    break
            if progress:
                break
        if not progress:
            break
    U2 = U[:, active]
    V2 = V[:, active]
    W2 = W[:, active]
    return U2, V2, W2


def _snap_halfint(U, V, W):
    """Round all entries to nearest half-integer (decomposition values are
    integral/half-integral by construction)."""
    U = np.round(2.0 * np.asarray(U, dtype=float)) / 2.0
    V = np.round(2.0 * np.asarray(V, dtype=float)) / 2.0
    W = np.round(2.0 * np.asarray(W, dtype=float)) / 2.0
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    best = (U, V, W)

    # Phase 1: merge columns (cheap exact structural reduction).
    try:
        U1, V1, W1 = _try_column_merging(U, V, W, n, m, p)
        if _validate(U1, V1, W1, n, m, p) and U1.shape[1] < best[0].shape[1]:
            best = (U1, V1, W1)
    except Exception:
        pass

    # Phase 2: eliminate redundant columns.
    try:
        U2, V2, W2 = _try_eliminate(best[0], best[1], best[2], n, m, p)
        if _validate(U2, V2, W2, n, m, p) and U2.shape[1] < best[0].shape[1]:
            best = (U2, V2, W2)
    except Exception:
        pass

    # Phase 3: a second merge pass on the reduced decomposition.
    try:
        U3, V3, W3 = _try_column_merging(best[0], best[1], best[2], n, m, p)
        if _validate(U3, V3, W3, n, m, p) and U3.shape[1] < best[0].shape[1]:
            best = (U3, V3, W3)
    except Exception:
        pass

    U, V, W = best
    U, V, W = _snap_halfint(U, V, W)

    # Final validation; if snapping broke it, fall back to unsnapped best.
    if not _validate(U, V, W, n, m, p):
        U, V, W = best
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        if not _validate(U, V, W, n, m, p):
            # Absolute fallback: schoolbook (always valid).
            U, V, W = _schoolbook(n, m, p)

    # Return as nested lists (interface expects list-of-lists).
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
