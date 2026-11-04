"""Rewritten program: exhaustive block-composition search with Strassen for 2x2x2 sub-blocks."""
DIMENSIONS = [3, 5, 5]
import time

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _compositions(total, min_part=1):
    """Yield all ordered compositions of `total` into parts >= min_part."""
    def rec(rem, max_part, prefix):
        if rem == 0:
            yield list(prefix)
            return
        for part in range(min_part, min(max_part, rem) + 1):
            prefix.append(part)
            yield from rec(rem - part, part, prefix)
            prefix.pop()
    yield from rec(total, total, [])

def _strassen_222():
    """Seven rank-1 tensors for 2x2x2 Strassen decomposition."""
    return [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

def _rank_of(n_splits, m_splits, p_splits):
    """Compute the rank for a given block composition (Strassen on 2x2x2, schoolbook elsewhere)."""
    r = 0
    for a in n_splits:
        for b in m_splits:
            for c in p_splits:
                if a == 2 and b == 2 and c == 2:
                    r += 7
                else:
                    r += a * b * c
    return r

def _build(n_splits, m_splits, p_splits, n, m, p):
    n_offsets, m_offsets, p_offsets = [0], [0], [0]
    for s in n_splits[:-1]: n_offsets.append(n_offsets[-1] + s)
    for s in m_splits[:-1]: m_offsets.append(m_offsets[-1] + s)
    for s in p_splits[:-1]: p_offsets.append(p_offsets[-1] + s)

    strassen = _strassen_222()
    U_entries, V_entries, W_entries = [], [], []
    rank = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                a, b, c = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                io, jo, ko = n_offsets[i_b], m_offsets[j_b], p_offsets[k_b]
                if a == 2 and b == 2 and c == 2:
                    for u_pat, v_pat, w_pat in strassen:
                        col = rank
                        for li, lj, val in u_pat:
                            U_entries.append(((io+li)*m + (jo+lj), col, val))
                        for lj, lk, val in v_pat:
                            V_entries.append(((jo+lj)*p + (ko+lk), col, val))
                        for lk, li, val in w_pat:
                            W_entries.append(((ko+lk)*n + (io+li), col, val))
                        rank += 1
                else:
                    for li in range(a):
                        for lj in range(b):
                            for lk in range(c):
                                col = rank
                                U_entries.append(((io+li)*m + (jo+lj), col, 1))
                                V_entries.append(((jo+lj)*p + (ko+lk), col, 1))
                                W_entries.append(((ko+lk)*n + (io+li), col, 1))
                                rank += 1
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for idx, col, val in U_entries: U[idx][col] = val
    for idx, col, val in V_entries: V[idx][col] = val
    for idx, col, val in W_entries: W[idx][col] = val
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    best_rank = None
    best_comp = None
    for n_s in _compositions(n):
        for m_s in _compositions(m):
            for p_s in _compositions(p):
                r = _rank_of(n_s, m_s, p_s)
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_comp = (n_s, m_s, p_s)
                if time.time() > _DEADLINE:
                    break
            if time.time() > _DEADLINE:
                break
        if time.time() > _DEADLINE:
            break
    if best_comp is None:
        best_comp = ([1,1,1], [2,2,1], [2,2,1])
    return _build(*best_comp, n, m, p)
# EVOLVE-BLOCK-END
