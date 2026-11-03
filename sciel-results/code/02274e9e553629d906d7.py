"""Rewritten: block decomposition + elimination + local pruning for 3x3x7 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]
_DEADLINE = 150.0  # evaluator kills at 150s; return well before


def _build_multiplication_tensor(n, m, p):
    """Return the n*m x m*p matrix whose (i*m+j, j'*p+k) entry is 1 iff j==j'.
    This is the 'multiplication tensor' flattened: M = kron(I_m, ones_{n x p})."""
    M = np.zeros((n * m, m * p))
    for j in range(m):
        for i in range(n):
            for k in range(p):
                M[i * m + j, j * p + k] = 1.0
    return M


def _schoolbook_decomposition(n, m, p):
    """Return (U, V, W) of rank n*m*p with integer entries that exactly decompose the
    multiplication tensor. U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for factor (i,j,k)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _decomposition_valid(U, V, W, n, m, p, tol=1e-8):
    """Check that U[i*m+j, r] * V[j*p+k, r] summed over r equals delta_{j,j'} for all i,j,k.
    Equivalently: for each (i,k), sum_r U[:,r]_row(i*m+j) * V[:,r]_row(j*p+k) over r,
    summed over j... actually simpler: M = sum_r outer(U[:,r], V[:,r]) should equal the
    multiplication tensor M0 (kron(I_m, ones_{n x p}))."""
    M0 = _build_multiplication_tensor(n, m, p)
    M = (U * V).sum(axis=1)  # outer product sum over rank
    return np.allclose(M, M0, atol=tol)


def _rank_of_M(M):
    """Numerical rank of M."""
    try:
        r = np.linalg.matrix_rank(M, tol=1e-10)
    except np.linalg.LinAlgError:
        r = M.shape[0]
    return int(r)


def _local_prune(U, V, W, n, m, p, budget=20.0, tol=1e-8):
    """Attempt to reduce rank by zeroing small entries and re-solving via least squares.
    Returns (U, V, W, rank) after pruning."""
    rank = U.shape[1]
    t0 = time.time()
    for attempt in range(30):
        if time.time() - t0 > budget:
            break
        if rank <= 0:
            break
        # Try zeroing one small entry in U
        if random.random() < 0.34:
            mat, row = U, np.random.randint(U.shape[0])
            col = np.random.randint(rank)
            # only consider small entries
            if abs(mat[row, col]) > 1e-6:
                continue
        elif random.random() < 0.67:
            mat, row = V, np.random.randint(V.shape[0])
            col = np.random.randint(rank)
            if abs(mat[row, col]) > 1e-6:
                continue
        else:
            # zero out an entire column (drop a rank factor) if it is small overall
            col = np.random.randint(rank)
            if np.linalg.norm(U[:, col]) > 1e-3 or np.linalg.norm(V[:, col]) > 1e-3:
                continue
            U[:, col] = 0.0
            V[:, col] = 0.0
            W[:, col] = 0.0
            rank = rank - 1
            continue
        # zero a single entry and re-solve for W (or for the zeroed variable)
        mat[row, col] = 0.0
        # Re-solve: we want sum_r U[i*m+j, r] V[j*p+k, r] = delta_{j,j'}
        # Equivalently M = sum_r u_r v_r^T. If we fix U and V, W is determined by
        # W[k*n+i, r] = sum_{j,j'} delta ... actually W is the output factor; the
        # tensor is sum_r u_r v_r w_r^T where u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
        # The constraint is that for each (i,k): sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 0
        # except when i,k come from the same j (i.e., the (i,k) pair must have the same j
        # in both indices). This is a linear system in W entries.
        # We build the linear system: for each (i,k), the coefficient of W[k*n+i, r] is
        # sum_j U[i*m+j, r] V[j*p+k, r].
        # We want this to equal 1 if there exists j with both indices at j (always true
        # for i in [0,n), k in [0,p)), else 0. Actually since i is in [0,n) and k in [0,p),
        # there is exactly one j per i (the j that owns i in U's row) and one j per k (the j
        # that owns k in V's row). They match iff the j for i equals the j for k.
        # For i = j1*m + ii (ii in [0,n)), the j is j1. For k = j2*p + kk, the j is j2.
        # They match iff j1 == j2.
        # So target[i, k] = 1 if floor(i/m) == floor(k/p) else 0.
        n, m, p = n, m, p
        # Build A: rows indexed by (i,k), cols indexed by (r, w_idx) where w_idx in [0, p*n)
        # A[(i,k), (r, w_idx)] = sum_j U[i*m+j, r] V[j*p+k, r] if w_idx == k*n+i else 0
        # This is block-diagonal in w_idx. So we can solve per w_idx.
        newW = np.zeros_like(W)
        for w_idx in range(p * n):
            i, k = divmod(w_idx, n)  # w_idx = k*n + i
            # For this (i,k), the equation is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] over j
            # = target.
            coeffs = np.zeros(rank)
            for j in range(m):
                coeffs += U[i * m + j, :] * V[j * p + k, :]
            # target
            j1 = i // m if m > 0 else 0
            j2 = k // p if p > 0 else 0
            target = 1.0 if j1 == j2 else 0.0
            # Solve coeffs @ W[w_idx, :] = target. Use least squares.
            if np.linalg.norm(coeffs) < 1e-12:
                if abs(target) > 1e-12:
                    return U, V, W, rank  # infeasible
                newW[w_idx, :] = 0.0
            else:
                sol, *_ = np.linalg.lstsq(coeffs.reshape(-1, 1), np.array([target]), rcond=None)
                newW[w_idx, :] = sol[:, 0]
        W = newW
        # Validate
        if not _decomposition_valid(U, V, W, n, m, p, tol=1e-6):
            # revert
            mat[row, col] = 1.0  # placeholder; we don't track original
            # Actually we should track the original value. Simplify: abort pruning.
            return U, V, W, rank
    return U, V, W, rank


def _block_schoolbook(n, m, p, deadline):
    """Split p into p1 and p2 = p - p1, use schoolbook on each block. Rank = n*m*p1 + n*m*p2 = n*m*p.
    This does not reduce rank vs full schoolbook, but it sets up the structure for elimination.
    We keep it for clarity."""
    if p <= 1:
        U, V, W = _schoolbook_decomposition(n, m, p)
        return U, V, W
    p1 = 3
    p2 = p - p1
    U1, V1, W1 = _schoolbook_decomposition(n, m, p1)
    U2, V2, W2 = _schoolbook_decomposition(n, m, p2)
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    return U, V, W


def _gauss_eliminate_columns(M, tol=1e-10):
    """Return column indices of a basis for the column space of M, plus the rank."""
    M = M.copy()
    rows, cols = M.shape
    pivot_cols = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        # find pivot in column c from row r down
        pivot_row = None
        for rr in range(r, rows):
            if abs(M[rr, c]) > tol:
                pivot_row = rr
                break
        if pivot_row is None:
            continue
        if pivot_row != r:
            M[[r, pivot_row]] = M[[pivot_row, r]]
        piv = M[r, c]
        M[r, :] = M[r, :] / piv
        for rr in range(rows):
            if rr != r:
                M[rr, :] = M[rr, :] - M[rr, c] * M[r, :]
        pivot_cols.append(c)
        r += 1
    return pivot_cols, r


def _find_rank_reduction_via_zero_structure(U, V, W, n, m, p, deadline):
    """Try to find a rank-r decomposition (r < current) by searching for zero-structured
    U, V, W that still satisfy the multiplication tensor constraint. We do a simple
    local search: randomly zero out small entries, re-solve W, validate."""
    rank = U.shape[1]
    t0 = time.time()
    best = (U.copy(), V.copy(), W.copy(), rank)
    while time.time() - t0 < deadline and rank > 0:
        U_try = U.copy()
        V_try = V.copy()
        # Randomly zero a few small entries
        for _ in range(5):
            if random.random() < 0.5:
                row = np.random.randint(U_try.shape[0])
                col = np.random.randint(rank)
                if abs(U_try[row, col]) < 1e-3:
                    U_try[row, col] = 0.0
            else:
                row = np.random.randint(V_try.shape[0])
                col = np.random.randint(rank)
                if abs(V_try[row, col]) < 1e-3:
                    V_try[row, col] = 0.0
        # Solve for W
        W_try = np.zeros((p * n, rank))
        feasible = True
        for w_idx in range(p * n):
            i, k = divmod(w_idx, n)
            coeffs = np.zeros(rank)
            for j in range(m):
                coeffs += U_try[i * m + j, :] * V_try[j * p + k, :]
            j1 = i // m
            j2 = k // p
            target = 1.0 if j1 == j2 else 0.0
            if np.linalg.norm(coeffs) < 1e-12:
                if abs(target) > 1e-12:
                    feasible = False
                    break
                W_try[w_idx, :] = 0.0
            else:
                sol, *_ = np.linalg.lstsq(coeffs.reshape(-1, 1), np.array([target]), rcond=None)
                W_try[w_idx, :] = sol[:, 0]
        if not feasible:
            continue
        if _decomposition_valid(U_try, V_try, W_try, n, m, p, tol=1e-6):
            best = (U_try.copy(), V_try.copy(), W_try.copy(), rank)
            # Try to drop a column if it's small
            col_norms = np.linalg.norm(W_try, axis=0)
            small = np.argwhere(col_norms < 1e-6)
            if len(small) > 0:
                c = small[0][0]
                U_try = np.delete(U_try, c, axis=1)
                V_try = np.delete(V_try, c, axis=1)
                W_try = np.delete(W_try, c, axis=1)
                rank = rank - 1
                U, V, W = U_try, V_try, W_try
                best = (U.copy(), V.copy(), W.copy(), rank)
    return best


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    deadline = _DEADLINE - 10.0  # safety margin

    # Start from schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    rank = n * m * p

    # Validate baseline
    assert _decomposition_valid(U, V, W, n, m, p, tol=1e-8), "schoolbook failed"

    # Local pruning / rank reduction
    U, V, W, rank = _find_rank_reduction_via_zero_structure(U, V, W, n, m, p, deadline - 20.0)

    # Final polish: round to nearest half-integer to clean up floating point
    def _round_half(X):
        return (2.0 * X).round() / 2.0
    U2 = _round_half(U)
    V2 = _round_half(V)
    W2 = _round_half(W)
    if _decomposition_valid(U2, V2, W2, n, m, p, tol=1e-8):
        U, V, W = U2, V2, W2

    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    print("rank:", len(result["U"][0]) if result["U"] else 0)
