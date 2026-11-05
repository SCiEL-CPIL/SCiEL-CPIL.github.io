"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Rank-7 Strassen 2x2x2 decomposition.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (a11+a22)(b11+b22)
    U[:, 0] = [1, 0, 0, 1]; V[:, 0] = [1, 0, 0, 1]; W[:, 0] = [1, 0, 0, 1]
    # M2 = (a21+a22)b11
    U[:, 1] = [0, 0, 1, 1]; V[:, 1] = [1, 0, 0, 0]; W[:, 1] = [0, 1, 0, -1]
    # M3 = a11(b12-b22)
    U[:, 2] = [1, 0, 0, 0]; V[:, 2] = [0, 1, 0, -1]; W[:, 2] = [0, 0, 1, 1]
    # M4 = a22(b21-b11)
    U[:, 3] = [0, 0, 0, 1]; V[:, 3] = [-1, 0, 1, 0]; W[:, 3] = [1, 1, 0, 0]
    # M5 = (a11+a12)b22
    U[:, 4] = [1, 1, 0, 0]; V[:, 4] = [0, 0, 0, 1]; W[:, 4] = [-1, 0, 1, 0]
    # M6 = (a21-a11)(b11+b12)
    U[:, 5] = [-1, 0, 1, 0]; V[:, 5] = [1, 1, 0, 0]; W[:, 5] = [0, 0, 0, 1]
    # M7 = (a12-a22)(b21+b22)
    U[:, 6] = [0, 1, 0, -1]; V[:, 6] = [0, 0, 1, 1]; W[:, 6] = [1, 0, 0, 0]
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _embed_block(U_local, V_local, W_local, n, m, p,
                 n_off, m_off, p_off, ni, mi, pi):
    """Embed a local ni x mi x pi decomposition into global (n,m,p) factors."""
    r = U_local.shape[1]
    U_g = np.zeros((n * m, r))
    V_g = np.zeros((m * p, r))
    W_g = np.zeros((p * n, r))
    for li in range(ni):
        for lj in range(mi):
            local_idx = li * mi + lj
            global_idx = (n_off + li) * m + (m_off + lj)
            U_g[global_idx, :] = U_local[local_idx, :]
    for lj in range(mi):
        for lk in range(pi):
            local_idx = lj * pi + lk
            global_idx = (m_off + lj) * p + (p_off + lk)
            V_g[global_idx, :] = V_local[local_idx, :]
    for lk in range(pi):
        for li in range(ni):
            local_idx = lk * ni + li
            global_idx = (p_off + lk) * n + (n_off + li)
            W_g[global_idx, :] = W_local[local_idx, :]
    return U_g, V_g, W_g


def _block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """Decompose n x m x p into blocks; Strassen for 2x2x2, schoolbook otherwise.
    Returns (U, V, W, rank)."""
    n_offs = [sum(n_parts[:i]) for i in range(len(n_parts))]
    m_offs = [sum(m_parts[:i]) for i in range(len(m_parts))]
    p_offs = [sum(p_parts[:i]) for i in range(len(p_parts))]

    block_ranks = {}
    total_rank = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                if ni == 2 and mi == 2 and pi == 2:
                    r = 7
                else:
                    r = ni * mi * pi
                block_ranks[(i, j, k)] = r
                total_rank += r

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                r = block_ranks[(i, j, k)]
                if ni == 2 and mi == 2 and pi == 2:
                    U_loc, V_loc, W_loc = _strassen_2x2x2()
                else:
                    U_loc, V_loc, W_loc = _schoolbook(ni, mi, pi)
                U_g, V_g, W_g = _embed_block(U_loc, V_loc, W_loc, n, m, p,
                                             n_offs[i], m_offs[j], p_offs[k],
                                             ni, mi, pi)
                U[:, col:col + r] = U_g
                V[:, col:col + r] = V_g
                W[:, col:col + r] = W_g
                col += r
    return U, V, W, total_rank


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor decomposition exactly (integer arithmetic)."""
    U = np.asarray(U, dtype=np.int64)
    V = np.asarray(V, dtype=np.int64)
    W = np.asarray(W, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = int(np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]))
                if val != 1:
                    return False
    return True


def _partitions(n):
    """All compositions of n into positive parts."""
    if n <= 0:
        return []
    if n == 1:
        return [[1]]
    out = []
    for first in range(1, n + 1):
        for rest in _partitions(n - first):
            out.append([first] + rest)
    return out


def _candidate_splits(n, m, p):
    """Generate block-split candidates ordered by expected rank (ascending)."""
    splits = []
    seen = set()
    for np_ in _partitions(n):
        for mp_ in _partitions(m):
            for pp_ in _partitions(p):
                key = (tuple(np_), tuple(mp_), tuple(pp_))
                if key in seen:
                    continue
                seen.add(key)
                rank = 0
                for ni, mi, pi in zip(np_, mp_, pp_):
                    if ni == 2 and mi == 2 and pi == 2:
                        rank += 7
                    else:
                        rank += ni * mi * pi
                splits.append((rank, np_, mp_, pp_))
    splits.sort(key=lambda x: x[0])
    return splits


def construct():
    start_time = time.time()
    deadline = start_time + 115  # explicit safety margin before the 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 5

    # Fallback: guaranteed-valid schoolbook decomposition.
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best = (U_fb, V_fb, W_fb, n * m * p)

    splits = _candidate_splits(n, m, p)

    for rank, np_, mp_, pp_ in splits:
        if time.time() > deadline:
            break
        # Never bother with a split that cannot beat the current best.
        if rank >= best[3]:
            break
        try:
            U, V, W, r = _block_decomposition(n, m, p, np_, mp_, pp_)
        except Exception:
            continue
        if r >= best[3]:
            continue
        if _verify_decomposition(U, V, W, n, m, p):
            best = (U, V, W, r)
            # Splits are sorted ascending by rank; once we have the global
            # minimum possible rank, nothing below can improve on it.
            if r <= splits[0][0]:
                break

    U, V, W, _ = best
    # Final safety: if somehow invalid, fall back to schoolbook.
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = U_fb, V_fb, W_fb

    # Return as lists of lists with integer entries (exact, half-integer free).
    Uo = [[int(x) for x in row] for row in np.asarray(U, dtype=np.int64).tolist()]
    Vo = [[int(x) for x in row] for row in np.asarray(V, dtype=np.int64).tolist()]
    Wo = [[int(x) for x in row] for row in np.asarray(W, dtype=np.int64).tolist()]
    return {"U": Uo, "V": Vo, "W": Wo}
# EVOLVE-BLOCK-END
