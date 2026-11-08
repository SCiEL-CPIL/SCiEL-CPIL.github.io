"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _naive_decomp(n, m, p):
    """Build the naive rank = n*m*p decomposition.

    For each (i,j,k): C[i,k] += A[i,j] * B[j,k]
    U row i*m+j gets 1, V row j*p+k gets 1, W row k*n+i gets 1.
    """
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for j in range(m):
        for i in range(n):
            for k in range(p):
                c = i * m + j          # A coordinate
                d = j * p + k          # B coordinate
                e = k * n + i          # C coordinate (W[k*n+i])
                idx = j * (n * p) + i * p + k
                U[c, idx] = 1.0
                V[d, idx] = 1.0
                W[e, idx] = 1.0
    return U, V, W


def _drop_column(U, V, W, col, keep_cols):
    """Try to drop column `col` by expressing W[:,col] as a combination of
    W[:,keep_cols]. Returns (new_U, new_V, new_W, new_cols) or None."""
    Wc = W[:, col]
    Wk = W[:, keep_cols]
    # solve Wk @ a = Wc
    try:
        a, res, rank, sv = np.linalg.lstsq(Wk, Wc, rcond=None)
    except Exception:
        return None
    # check it's an exact solution
    if not np.allclose(Wk @ a, Wc, atol=1e-7, rtol=1e-7):
        return None
    new_U = np.zeros((U.shape[0], len(keep_cols)))
    new_V = np.zeros((V.shape[0], len(keep_cols)))
    new_W = np.zeros((W.shape[0], len(keep_cols)))
    for li, kc in enumerate(keep_cols):
        new_U[:, li] = U[:, kc] + a[li] * U[:, col]
        new_V[:, li] = V[:, kc] + a[li] * V[:, col]
        new_W[:, li] = W[:, kc] + a[li] * W[:, col]
    return new_U, new_V, new_W, list(keep_cols)


def _local_improve(U, V, W, time_budget):
    """Attempt to reduce rank via null-space column elimination."""
    start = time.time()
    cur_U, cur_V, cur_W = U.copy(), V.copy(), W.copy()
    cur_cols = list(range(cur_U.shape[1]))
    improved = True
    while improved:
        improved = False
        if time.time() - start > time_budget:
            break
        # find the most "redundant" column: smallest norm or check each
        # For efficiency, iterate columns and attempt drop when its W is
        # in span of others. Use a heuristic ordering.
        for col in list(cur_cols):
            if time.time() - start > time_budget:
                break
            keep_cols = [c for c in cur_cols if c != col]
            if not keep_cols:
                continue
            res = _drop_column(cur_U, cur_V, cur_W, col, keep_cols)
            if res is not None:
                cur_U, cur_V, cur_W, cur_cols = res
                improved = True
                break
    return cur_U, cur_V, cur_W, len(cur_cols)


def _to_int_matrices(U, V, W, cols):
    """Convert numpy float matrices (possibly with small floats) to int lists.
    Rounds to nearest integer; assumes entries are integer/half-integer."""
    U = np.rint(U).astype(int)
    V = np.rint(V).astype(int)
    W = np.rint(W).astype(int)
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 130.0  # safety margin under 150s kill

    # Step 1: naive decomposition (rank 60)
    U, V, W = _naive_decomp(n, m, p)
    rank = U.shape[1]
    best = (rank, U.copy(), V.copy(), W.copy())

    # Step 2: local improvement
    if time.time() < DEADLINE:
        time_budget = max(1.0, DEADLINE - time.time() - 5.0)
        try:
            nU, nV, nW, new_rank = _local_improve(U, V, W, time_budget)
            if new_rank < best[0]:
                best = (new_rank, nU, nV, nW)
        except Exception:
            pass

    r, U, V, W = best
    U, V, W = _to_int_matrices(U, V, W, None)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
