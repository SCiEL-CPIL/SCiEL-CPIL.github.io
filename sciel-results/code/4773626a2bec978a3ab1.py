"""Tensor decomposition for 2x3 matrix multiplication (A@B, B is 3x10).

Combines:
  * the explicit rank-55 base decomposition (current program's structure), and
  * exact rational rank reduction by flattening the tensor and extracting a
    minimal basis (inspiration 2's refinement machinery).

The flattened matrix has columns = rank-1 tensors (flattened to length 60).
Its column rank equals the rank of the decomposition. A minimal exact basis
of those columns gives the smallest rank among all decompositions that are
linear combinations of the base rank-1 tensors.
"""
import time
from fractions import Fraction
import numpy as np

def _build_base_decomposition():
    """Build the rank-55 base decomposition. Returns U, V, W as numpy arrays."""
    n, m, p = 2, 3, 10
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1

        # Term 1: (A00+A11)*(B0k0+B1k1) -> C0k0, C1k1
        U[0, col] += 1; U[4, col] += 1
        V[0*10+k0, col] += 1; V[1*10+k1, col] += 1
        W[k0*2+0, col] += 1; W[k1*2+1, col] += 1
        col += 1

        # Term 2: (A10+A11)*B0k0 -> C1k0, C1k1
        U[3, col] += 1; U[4, col] += 1
        V[0*10+k0, col] += 1
        W[k0*2+1, col] += 1; W[k1*2+1, col] -= 1
        col += 1

        # Term 3: A00*(B0k1-B1k1) -> C0k1, C1k1
        U[0, col] += 1
        V[0*10+k1, col] += 1; V[1*10+k1, col] -= 1
        W[k1*2+0, col] += 1; W[k1*2+1, col] += 1
        col += 1

        # Term 4: A11*(B1k0-B0k0) -> C0k0, C1k0
        U[4, col] += 1
        V[1*10+k0, col] += 1; V[0*10+k0, col] -= 1
        W[k0*2+0, col] += 1; W[k0*2+1, col] += 1
        col += 1

        # Term 5: (A00+A01)*B1k1 -> C0k0(-1), C0k1
        U[0, col] += 1; U[1, col] += 1
        V[1*10+k1, col] += 1
        W[k0*2+0, col] -= 1; W[k1*2+0, col] += 1
        col += 1

        # Term 6: (A10-A00)*(B0k0+B0k1) -> C1k1
        U[3, col] += 1; U[0, col] -= 1
        V[0*10+k0, col] += 1; V[0*10+k1, col] += 1
        W[k1*2+1, col] += 1
        col += 1

        # Term 7: (A01-A11)*(B1k0+B1k1) -> C0k0
        U[1, col] += 1; U[4, col] -= 1
        V[1*10+k0, col] += 1; V[1*10+k1, col] += 1
        W[k0*2+0, col] += 1
        col += 1

        # Terms 8-11: direct terms for row 2 of B
        U[2, col] += 1; V[2*10+k0, col] += 1; W[k0*2+0, col] += 1
        col += 1
        U[2, col] += 1; V[2*10+k1, col] += 1; W[k1*2+0, col] += 1
        col += 1
        U[5, col] += 1; V[2*10+k0, col] += 1; W[k0*2+1, col] += 1
        col += 1
        U[5, col] += 1; V[2*10+k1, col] += 1; W[k1*2+1, col] += 1
        col += 1

    return U, V, W, rank

def _verify_decomposition(U, V, W, n=2, m=3, p=10, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _flatten_tensor(U, V, W, n, m, p):
    """Flatten each rank-1 tensor (col r) to a vector of length n*m*p.

    Indexing: flat[(i*m+j) * (m*p) + (j*p+k)]  = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    """
    N = n * m * p
    R = U.shape[1]
    flat = np.zeros((N, R))
    for r in range(R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    flat[(i*m+j)*(m*p) + (j*p+k), r] = \
                        U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    return flat

def _exact_column_rank_and_basis(flat_int):
    """Compute the exact column rank of an integer matrix and a minimal exact
    basis (list of column indices) using Gaussian elimination over the rationals.

    flat_int: 2D numpy array of ints, shape (N, R).
    Returns (rank, basis_indices).
    """
    N, R = flat_int.shape
    rows = [list(map(Fraction, flat_int[i].tolist())) for i in range(N)]
    pivots = []          # list of (pivot_row, col)
    basis = []           # column indices of pivot columns (a minimal basis)
    for c in range(R):
        # find a row >= len(pivots) with nonzero in column c
        pr = len(pivots)
        found = -1
        for r in range(pr, N):
            if rows[r][c] != 0:
                found = r
                break
        if found < 0:
            continue
        if found != pr:
            rows[pr], rows[found] = rows[found], rows[pr]
        pv = rows[pr][c]
        # eliminate this column from all other rows
        for r in range(N):
            if r == pr:
                continue
            if rows[r][c] != 0:
                f = rows[r][c] / pv
                row = rows[r]
                prow = rows[pr]
                for cc in range(c, R):
                    row[cc] -= f * prow[cc]
        pivots.append((pr, c))
        basis.append(c)
    return len(basis), basis

def construct():
    _t0 = time.time()
    n, m, p = 2, 3, 10
    deadline = _t0 + 110  # safety margin before the 150s kill

    # Build base rank-55 decomposition
    U, V, W, rank = _build_base_decomposition()

    # Verify base decomposition (should hold)
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: return the base decomposition as-is
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank

    # --- Exact rank reduction via flattening ---
    # The flattened tensor's column rank is the rank of this decomposition.
    # Extract a minimal exact basis of the columns -> fewer rank-1 tensors.
    try:
        flat = _flatten_tensor(U, V, W, n, m, p)
        # Entries are products of small integers -> exact integers.
        flat_int = np.rint(flat).astype(object)
        # Convert to a plain int matrix for Fraction (avoid numpy object quirks)
        flat_int_list = [[int(v) for v in flat_int[i, :]] for i in range(flat_int.shape[0])]
        new_rank, basis = _exact_column_rank_and_basis(flat_int_list)

        if new_rank < rank:
            U2 = U[:, basis]
            V2 = V[:, basis]
            W2 = W[:, basis]
            if _verify_decomposition(U2, V2, W2, n, m, p):
                best_U, best_V, best_W, best_rank = U2.copy(), V2.copy(), W2.copy(), new_rank
    except Exception:
        pass

    # --- Secondary pass: proportional-column elimination on the reduced set ---
    # (cheap, exact, in case the basis extraction left any scalar-duplicate columns)
    cur_U, cur_V, cur_W, cur_rank = best_U.copy(), best_V.copy(), best_W.copy(), best_rank
    for _ in range(cur_rank + 2):
        if time.time() > deadline:
            break
        removed = False
        for col_idx in range(cur_rank):
            redundant = False
            for other_idx in range(cur_rank):
                if other_idx == col_idx:
                    continue
                U1, U2 = cur_U[:, col_idx], cur_U[:, other_idx]
                V1, V2 = cur_V[:, col_idx], cur_V[:, other_idx]
                W1, W2 = cur_W[:, col_idx], cur_W[:, other_idx]
                nz = np.abs(U2) > 1e-10
                if np.any(nz):
                    a_cand = U1[nz] / U2[nz]
                    if np.allclose(a_cand, a_cand[0], atol=1e-10):
                        a = a_cand[0]
                        if np.allclose(V1, a * V2, atol=1e-10) and \
                           np.allclose(W1, a * W2, atol=1e-10):
                            redundant = True
                            break
                if redundant:
                    break
            if redundant:
                mask = [i for i in range(cur_rank) if i != col_idx]
                cur_U = cur_U[:, mask]
                cur_V = cur_V[:, mask]
                cur_W = cur_W[:, mask]
                cur_rank -= 1
                if _verify_decomposition(cur_U, cur_V, cur_W, n, m, p):
                    best_U, best_V, best_W, best_rank = cur_U.copy(), cur_V.copy(), cur_W.copy(), cur_rank
                removed = True
                break
        if not removed:
            break

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
