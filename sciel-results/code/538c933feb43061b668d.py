"""Hybrid tiling with exact verification for 3x5 x 5x7 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

_STRASSEN = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
]

def _partitions(dim, max_size=None):
    """All ordered compositions of dim into parts of size <= max_size."""
    if max_size is None:
        max_size = dim
    if dim <= 0:
        return [()]
    results = []
    def rec(remaining, acc):
        if remaining == 0:
            results.append(tuple(acc))
            return
        for s in range(1, min(max_size, remaining) + 1):
            acc.append(s)
            rec(remaining - s, acc)
            acc.pop()
    rec(dim, [])
    return results

def _rank_of_block(si, sj, sk):
    """Rank for a single block of size si x sj x sk."""
    if si == 2 and sj == 2 and sk == 2:
        return 7
    return si * sj * sk

def _rank_of_packing(gi, gj, gk):
    """Total rank for a given tiling."""
    rank = 0
    for si in gi:
        for sj in gj:
            for sk in gk:
                rank += _rank_of_block(si, sj, sk)
    return rank

def _build_decomposition(gi, gj, gk, n, m, p):
    """Build full-size U, V, W for a given tiling."""
    n_off = [0]
    for s in gi:
        n_off.append(n_off[-1] + s)
    m_off = [0]
    for s in gj:
        m_off.append(m_off[-1] + s)
    p_off = [0]
    for s in gk:
        p_off.append(p_off[-1] + s)

    rank = 0
    terms = []
    for ni, si in enumerate(gi):
        for mi, sj in enumerate(gj):
            for pi, sk in enumerate(gk):
                r = _rank_of_block(si, sj, sk)
                terms.append((ni, mi, pi, si, sj, sk, r))
                rank += r

    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    col = 0
    for ni, mi, pi, si, sj, sk, r in terms:
        n0 = n_off[ni]
        m0 = m_off[mi]
        p0 = p_off[pi]

        if si == 2 and sj == 2 and sk == 2:
            for rr in range(7):
                af, bf, cf = _STRASSEN[rr]
                for (ar, ac, c) in af:
                    U[(n0+ar)*m + (m0+ac), col] += c
                for (br, bc, c) in bf:
                    V[(m0+br)*p + (p0+bc), col] += c
                for (cr, cc, c) in cf:
                    W[(p0+cc)*n + (n0+cr), col] += c
                col += 1
        else:
            for ii in range(si):
                for jj in range(sj):
                    for kk in range(sk):
                        U[(n0+ii)*m + (m0+jj), col] = 1
                        V[(m0+jj)*p + (p0+kk), col] = 1
                        W[(p0+kk)*n + (n0+ii), col] = 1
                        col += 1

    return U, V, W, rank

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Vectorized verification."""
    i_idx = np.repeat(np.arange(n), m*p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n*m)

    a_idx = i_idx * m + j_idx
    b_idx = j_idx * p + k_idx
    c_idx = k_idx * n + i_idx

    Ua = U[a_idx, :]
    Vb = V[b_idx, :]
    Wc = W[c_idx, :]
    prod = Ua * Vb * Wc
    s = prod.sum(axis=1)
    expected = np.ones(len(s))
    return np.all(np.abs(s - expected) <= tol)

def _try_packing(gi, gj, gk, n, m, p):
    """Try a specific packing and return (U,V,W,rank) if valid, else None."""
    U, V, W, rank = _build_decomposition(gi, gj, gk, n, m, p)
    if _verify_fast(U, V, W, n, m, p):
        return U, V, W, rank
    return None

def _search_packings(n, m, p, deadline):
    """Search over tilings sorted by predicted rank."""
    parts_n = _partitions(n, 4)
    parts_m = _partitions(m, 4)
    parts_p = _partitions(p, 4)

    candidates = []
    for gi in parts_n:
        for gj in parts_m:
            for gk in parts_p:
                r = _rank_of_packing(gi, gj, gk)
                candidates.append((r, gi, gj, gk))
    candidates.sort(key=lambda x: x[0])

    for r, gi, gj, gk in candidates:
        if time.time() > deadline:
            break
        result = _try_packing(gi, gj, gk, n, m, p)
        if result is not None:
            return result
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W, rank

def construct():
    start = time.time()
    deadline = start + 130.0  # Safety margin before 150s kill
    n, m, p = DIMENSIONS

    best = _search_packings(n, m, p, deadline)

    if best is None:
        U, V, W, rank = _schoolbook(n, m, p)
    else:
        U, V, W, rank = best

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
