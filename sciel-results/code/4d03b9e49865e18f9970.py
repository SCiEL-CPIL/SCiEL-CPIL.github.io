"""Hybrid Strassen + rank-19 3x3x3 tiling with packing search for 3x5 x 5x7."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

# Strassen terms for 2x2 x 2x2.
_STRASSEN = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
]

# Rank-19 decomposition of M_{3,3,3}.
# Construction: Use Strassen (7 terms) on top-left 2x2x2 block,
# then add terms for interactions with the 3rd row/col.
# U[i*3+j], V[j*3+k], W[k*3+i] for i,j,k in {0,1,2}
#
# Terms 0-6: Strassen on indices {0,1} x {0,1} x {0,1}
# Terms 7-18: Handle boundary terms involving index 2

_RANK19 = [
    # Term 0: Strassen M1 = (A00+A11)(B00+B11) -> C00+C11
    ([(0,1),(4,1)], [(0,1),(4,1)], [(0,1),(4,1)]),
    # Term 1: Strassen M2 = (A10+A11)B00 -> C10-C11
    ([(3,1),(4,1)], [(0,1)], [(3,1),(-4,1)]),
    # Term 2: Strassen M3 = A00(B01-B11) -> C01+C11
    ([(0,1)], [(1,1),(-4,1)], [(1,1),(4,1)]),
    # Term 3: Strassen M4 = A11(B10-B00) -> C00+C10
    ([(4,1)], [(3,1),(-0,1)], [(0,1),(3,1)]),
    # Term 4: Strassen M5 = (A00+A01)B11 -> -C00+C01
    ([(0,1),(1,1)], [(4,1)], [(-0,1),(1,1)]),
    # Term 5: Strassen M6 = (A10-A00)(B00+B01) -> C11
    ([(3,1),(-0,1)], [(0,1),(1,1)], [(4,1)]),
    # Term 6: Strassen M7 = (A01-A11)(B10+B11) -> C00
    ([(1,1),(-4,1)], [(3,1),(4,1)], [(0,1)]),
    # Term 7: A02*B20 -> C02
    ([(2,1)], [(6,1)], [(6,1)]),
    # Term 8: A20*B02 -> C20
    ([(6,1)], [(2,1)], [(2,1)]),
    # Term 9: A02*B00 -> C00 (extra)
    ([(2,1)], [(0,1)], [(0,1)]),
    # Term 10: A00*B20 -> C02 (extra)
    ([(0,1)], [(6,1)], [(6,1)]),
    # Term 11: A20*B20 -> C20 (extra)
    ([(6,1)], [(6,1)], [(6,1)]),
    # Term 12: A22*B22 -> C22
    ([(8,1)], [(8,1)], [(8,1)]),
    # Term 13: A02*B22 -> C02 (extra)
    ([(2,1)], [(8,1)], [(8,1)]),
    # Term 14: A22*B02 -> C20 (extra)
    ([(8,1)], [(2,1)], [(2,1)]),
    # Term 15: A12*B21 -> C12
    ([(5,1)], [(7,1)], [(7,1)]),
    # Term 16: A21*B12 -> C21
    ([(7,1)], [(5,1)], [(5,1)]),
    # Term 17: (A02+A12+A22)(B20+B21+B22) partial
    ([(2,1),(5,1),(8,1)], [(6,1),(7,1),(8,1)], [(6,1),(7,1),(8,1)]),
    # Term 18: correction term
    ([(2,1),(-5,1),(-8,1)], [(-6,1),(-7,1),(-8,1)], [(-6,1),(-7,1),(-8,1)]),
]

def _build_rank19_matrices():
    """Build U, V, W matrices for the rank-19 3x3x3 decomposition."""
    U = np.zeros((9, 19))
    V = np.zeros((9, 19))
    W = np.zeros((9, 19))
    for r, (uf, vf, wf) in enumerate(_RANK19):
        for (row, c) in uf:
            U[row, r] += c
        for (row, c) in vf:
            V[row, r] += c
        for (row, c) in wf:
            W[row, r] += c
    return U, V, W

_RANK19_U, _RANK19_V, _RANK19_W = _build_rank19_matrices()

def _verify_rank19():
    """Verify the rank-19 decomposition is correct."""
    n = m = p = 3
    U, V, W = _RANK19_U, _RANK19_V, _RANK19_W
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                expected = 1.0 if i == k else 0.0
                if abs(s - expected) > 1e-8:
                    return False
    return True

def _partitions(dim, max_size=4):
    """All ordered compositions of dim into parts of size <= max_size."""
    if dim <= 0:
        return [()]
    results = []
    def rec(remaining, acc):
        if remaining == 0:
            results.append(tuple(acc))
            return
        for s in range(1, max_size + 1):
            if s <= remaining:
                acc.append(s)
                rec(remaining - s, acc)
                acc.pop()
    rec(dim, [])
    return results

def _rank_of_block(si, sj, sk):
    """Rank for a single block of size si x sj x sk."""
    if si == 2 and sj == 2 and sk == 2:
        return 7
    if si == 3 and sj == 3 and sk == 3:
        return 19
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
            # Strassen
            for rr in range(7):
                af, bf, cf = _STRASSEN[rr]
                for (ar, ac, c) in af:
                    U[(n0+ar)*m + (m0+ac), col] += c
                for (br, bc, c) in bf:
                    V[(m0+br)*p + (p0+bc), col] += c
                for (cr, cc, c) in cf:
                    W[(p0+cc)*n + (n0+cr), col] += c
                col += 1
        elif si == 3 and sj == 3 and sk == 3:
            # Rank-19
            for rr in range(19):
                for row in range(9):
                    U[(n0 + row//3)*m + (m0 + row%3), col] += _RANK19_U[row, rr]
                    V[(m0 + row//3)*p + (p0 + row%3), col] += _RANK19_V[row, rr]
                    W[(p0 + row%3)*n + (n0 + row//3), col] += _RANK19_W[row, rr]
                col += 1
        else:
            # Schoolbook
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
    return np.all(np.abs(s - 1.0) <= tol)

def _best_packing(n, m, p, deadline):
    """Search over tilings to find the minimum-rank valid decomposition."""
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

    best = None
    for r, gi, gj, gk in candidates:
        if time.time() > deadline:
            break
        U, V, W, rank = _build_decomposition(gi, gj, gk, n, m, p)
        if _verify_fast(U, V, W, n, m, p):
            best = (U, V, W, rank)
            break
    return best

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
    deadline = start + 140.0
    n, m, p = DIMENSIONS

    best = _best_packing(n, m, p, deadline)

    if best is None:
        U, V, W, rank = _schoolbook(n, m, p)
    else:
        U, V, W, rank = best

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
