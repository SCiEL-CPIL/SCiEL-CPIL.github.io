import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
DEADLINE_MARGIN = 12.0   # stop work ~12s before the 120s hard limit


def _elapsed():
    return time.time() - _START_TIME


def _within_budget():
    return _elapsed() < (120.0 - DEADLINE_MARGIN)


def _schoolbook_block(n, m, p):
    """Return (U, V, W) for the n x m by m x p multiplication tensor,
    indexed with the official trace convention:
        U index (i, j)  -> i*m + j
        V index (j, k)  -> j*p + k
        W index (k, i)  -> k*n + i
    Each rank slot (i,j,k) sets U[i,j]=1, V[j,k]=1, W[k,i]=1.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check that the decomposition is exact on random integer inputs.
    Returns True if all sampled products match.
    """
    rng = np.random.RandomState(20240517)
    trials = 3
    for _ in range(trials):
        A = rng.randint(-3, 4, size=(n, m))
        B = rng.randint(-3, 4, size=(m, p))
        C_expected = A @ B
        # Reconstruct via tensor: C = sum_r u_{i,j}^{(r)} v_{j,k}^{(r)} w_{k,i}^{(r)}
        U = np.asarray(U, dtype=object)
        V = np.asarray(V, dtype=object)
        W = np.asarray(W, dtype=object)
        rank = U.shape[1]
        # Build rank-slices
        # U[i*m+j], V[j*p+k], W[k*n+i]
        C = np.zeros((n, p), dtype=object)
        for r in range(rank):
            ur = U[:, r]          # length n*m
            vr = V[:, r]          # length m*p
            wr = W[:, r]          # length p*n
            Ur = ur.reshape(n, m)
            Vr = vr.reshape(m, p)
            Wr = wr.reshape(p, n)
            # contribution: C[i,k] += sum_j Ur[i,j] * Vr[j,k] * Wr[k,i]
            # Wr[k,i] depends on i, so compute per (i,k)
            # M = Ur @ Vr  (n x p); then C[i,k] += M[i,k] * Wr[k,i]
            M = Ur @ Vr           # (n x p)
            # Wr is (p x n): Wr[k,i]
            # Multiply elementwise M[i,k] * Wr[k,i]
            C += M * Wr.T
        if not np.array_equal(C, C_expected):
            return False
    return True


def _to_lists(arr):
    return [[int(x) if x == int(x) else x for x in row] for row in arr.tolist()]


def construct():
    """
    Returns a rank-minimal (within the schoolbook/tiled family) exact tensor
    decomposition for multiplying a 2x3 matrix by a 3x9 matrix.

    Construction: split the 2x3 x 3x9 multiplication into three independent
    2x3 x 3x3 tiles along the output columns. Each 2x3 x 3x3 tile uses the
    schoolbook (rank 6) decomposition. The tiles are block-diagonal in the
    rank index, giving total rank 3 * 6 = 18 per tile-set... actually each
    tile is rank 6, three tiles => rank 18? No: each 2x3x3 tile has rank
    2*3*3 = 18. Three tiles => 54 total.

    The block-diagonal assembly is verified on random inputs before returning.
    """
    n, m, p = DIMENSIONS

    # Decompose p into tiles of width t=3
    t = 3
    n_tiles = p // t

    # Total rank
    rank_per_tile = n * m * t
    total_rank = n_tiles * rank_per_tile

    U = np.zeros((n * m, total_rank), dtype=object)
    V = np.zeros((m * p, total_rank), dtype=object)
    W = np.zeros((p * n, total_rank), dtype=object)

    for tile in range(n_tiles):
        if not _within_budget():
            break
        # This tile handles output columns [tile*t, (tile+1)*t)
        # and input rows all m (the 3x3 tile is B[:, tile*t:(tile+1)*t])
        u_tile, v_tile, w_tile = _schoolbook_block(n, m, t)

        col_start = tile * rank_per_tile
        col_end = col_start + rank_per_tile

        # U: same for every tile (the A factor is shared)
        U[:, col_start:col_end] = u_tile

        # V: place v_tile into the right column block of B
        # v_tile has shape (m*t, rank_per_tile), indexing (j, k') where
        # k' in [0, t). B's column index for this tile is tile*t + k'.
        # v_tile row index j*p_tile + k' where p_tile = t.
        # But V must be indexed as (j, k) with k in [0, p).
        for j in range(m):
            for kp in range(t):
                # v_tile row: j * t + kp
                val = v_tile[j * t + kp, :]
                # V row: j * p + (tile*t + kp)
                V[j * p + (tile * t + kp), col_start:col_end] = val

        # W: w_tile has shape (t*n, rank_per_tile), indexing (k', i)
        # W must be indexed (k, i) with k in [0, p).
        # k = tile*t + kp
        for kp in range(t):
            for i in range(n):
                w_tile_row = kp * n + i
                W[(tile * t + kp) * n + i, col_start:col_end] = w_tile_row if False else w_tile[w_tile_row, :]

    # Verify before returning; if verification fails, fall back to full schoolbook.
    if _verify(U, V, W, n, m, p):
        return {"U": _to_lists(U), "V": _to_lists(V), "W": _to_lists(W)}

    # Fallback: full schoolbook (always correct)
    U, V, W = _schoolbook_block(n, m, p)
    return {"U": _to_lists(U), "V": _to_lists(V), "W": _to_lists(W)}
# EVOLVE-BLOCK-END
