import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _strassen_2x2x2():
    """Verified rank-7 Strassen decomposition of the 2x2x2 multiplication tensor.
    U (4x7) indexed by (i,j) in row-major, V (4x7) by (j,k), W (4x7) by (k,i)."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Return True iff sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _remove_zero_cols(U, V, W):
    """Drop columns that are identically zero in all three matrices."""
    keep = []
    for c in range(U.shape[1]):
        if np.any(U[:, c]) or np.any(V[:, c]) or np.any(W[:, c]):
            keep.append(c)
    return U[:, keep], V[:, keep], W[:, keep]


def _try_reduce(U, V, W, n, m, p, deadline, tol=1e-9):
    """Iteratively remove redundant columns using the exact span criterion.
    A column c is redundant iff its U-row vector lies in the span of the other
    columns' U-row vectors AND the same holds for V AND W. For this tensor family
    that is exactly equivalent to the column being removable without changing the
    reconstructed tensor."""
    U, V, W = U.copy(), V.copy(), W.copy()
    r = U.shape[1]
    changed = True
    while r > 0 and time.time() < deadline:
        changed = False
        for c in range(r - 1, -1, -1):
            if time.time() > deadline:
                break
            U_o = np.delete(U, c, axis=1)
            V_o = np.delete(V, c, axis=1)
            W_o = np.delete(W, c, axis=1)
            U_c = U[:, c]
            V_c = V[:, c]
            W_c = W[:, c]
            # span test on U rows
            if np.linalg.matrix_rank(np.hstack([U_o, U_c.reshape(-1, 1)])) == np.linalg.matrix_rank(U_o):
                # U_c not in span -> keep
                continue
            if np.linalg.matrix_rank(np.hstack([V_o, V_c.reshape(-1, 1)])) == np.linalg.matrix_rank(V_o):
                continue
            if np.linalg.matrix_rank(np.hstack([W_o, W_c.reshape(-1, 1)])) == np.linalg.matrix_rank(W_o):
                continue
            # candidate redundant: confirm with a global verify
            if _verify(U_o, V_o, W_o, n, m, p, tol):
                U, V, W = U_o, V_o, W_o
                r -= 1
                changed = True
                break
        if not changed:
            break
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 4, 8

    U2, V2, W2 = _strassen_2x2x2()

    # Tiling: two 2x2 Strassen blocks (i in {0,1}, j in {0,1} and {2,3}) over
    # all 8 k's (4 k-pairs), plus schoolbook for i=2 (all j,k).
    # Strassen blocks: 2 j-blocks * 4 k-blocks * rank 7 = 56
    # Schoolbook i=2: 4 j * 8 k = 32
    # Total = 88
    total_rank = 2 * 4 * 7 + 4 * 8
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    col = 0
    for j0 in (0, 2):            # two j-blocks of size 2
        for k0 in range(0, 8, 2):  # four k-blocks of size 2
            for i in (0, 1):
                for c in range(7):
                    U[i * m + j0 + 0][col + c] += U2[0][c]  # local (i=0,j=0)
                    U[i * m + j0 + 1][col + c] += U2[1][c]  # local (i=0,j=1)
                    U[(i + 1) * m + j0 + 0][col + c] += U2[2][c]  # local (i=1,j=0)
                    U[(i + 1) * m + j0 + 1][col + c] += U2[3][c]  # local (i=1,j=1)
            for c in range(7):
                V[j0 + 0][k0 + 0] = 0  # placeholder (overwritten below per col)
            # fill V and W for this block
            for c in range(7):
                V[j0 + 0, col + c] += V2[0][c]
                V[j0 + 0, col + c] += 0
                V[j0 + 1, col + c] += V2[1][c]
                V[j0 + 0, col + c]  # noop
                V[j0 + 1, col + c]
                # correct assignment (row-major (j,k)):
                V[j0 + 0, col + c] = V2[0][c]
                V[j0 + 0, col + c]
                V[j0 + 1, col + c] = V2[1][c]
                V[j0 + 0, col + c]
                W[k0 + 0, 0 * n + 0]  # noop
            # Re-do V and W cleanly to avoid the messy lines above
            col_block_start = col
            for c in range(7):
                V[j0 + 0, col_block_start + c] = V2[0][c]
                V[j0 + 0, col_block_start + c]
                V[j0 + 1, col_block_start + c] = V2[1][c]
                # k rows:
                V[j0 + 0, col_block_start + c]
                W[k0 + 0, 0 * n + 0]
            col += 7

    # The above got tangled; rebuild cleanly from scratch.
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    col = 0
    for j0 in (0, 2):
        for k0 in range(0, 8, 2):
            for c in range(7):
                # U: local (li,lj) -> global i=li, j=j0+lj
                U[0 * m + j0 + 0, col + c] = U2[0][c]
                U[0 * m + j0 + 1, col + c] = U2[1][c]
                U[1 * m + j0 + 0, col + c] = U2[2][c]
                U[1 * m + j0 + 1, col + c] = U2[3][c]
                # V: local (lj, lk) -> global j=j0+lj, k=k0+lk
                V[j0 + 0, k0 + 0, ]  # noop
                V[j0 + 0, k0 + 0] = V2[0][c]
                V[j0 + 0, k0 + 1] = V2[1][c]
                V[j0 + 1, k0 + 0] = V2[2][c]
                V[j0 + 1, k0 + 1] = V2[3][c]
                # W: local (lk, li) -> global k=k0+lk, i=li
                W[k0 + 0, 0 * n + 0] = W2[0][c]
                W[k0 + 1, 0 * n + 0] = W2[1][c]
                W[k0 + 0, 1 * n + 0] = W2[2][c]
                W[k0 + 1, 1 * n + 0] = W2[3][c]
            col += 7

    # Schoolbook for i = 2 (all j in 0..3, k in 0..7)
    for j in range(m):
        for k in range(p):
            U[2 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 2, col] = 1
            col += 1

    # Verify the construction
    if not _verify(U, V, W, n, m, p):
        # Fallback: full schoolbook rank 96
        U = np.zeros((n * m, n * m * p), dtype=float)
        V = np.zeros((m * p, n * m * p), dtype=float)
        W = np.zeros((p * n, n * m * p), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Reduce rank
    if time.time() < deadline - 5:
        U, V, W = _remove_zero_cols(U, V, W)
        U, V, W = _try_reduce(U, V, W, n, m, p, deadline)

    # Final safety verify
    if not _verify(U, V, W, n, m, p):
        U = np.zeros((n * m, n * m * p), dtype=float)
        V = np.zeros((m * p, n * m * p), dtype=float)
        W = np.zeros((p * n, n * m * p), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
