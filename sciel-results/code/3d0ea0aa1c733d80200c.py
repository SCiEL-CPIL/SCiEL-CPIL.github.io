"""Matrix multiplication 3x5x5 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0

    # Strassen patterns for 2x2x2
    # Each pattern is (U_entries, V_entries, W_entries)
    # U_entries: list of (local_i, local_j, value)
    # V_entries: list of (local_j, local_k, value)
    # W_entries: list of (local_k, local_i, value)
    # Convention: W[k*n+i, c]
    strassen_patterns = [
        # M1 = (a00+a11)(b00+b11) -> C00+=M1, C11+=M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (a10+a11)b00 -> C10+=M2, C11-=M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        # M3 = a00(b01-b11) -> C01+=M3, C11+=M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        # M4 = a11(b10-b00) -> C00+=M4, C10+=M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        # M5 = (a00+a01)b11 -> C00-=M5, C01+=M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
        # M6 = (a10-a00)(b00+b01) -> C11+=M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (a01-a11)(b10+b11) -> C00+=M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    def compute_rank(n_splits, m_splits, p_splits):
        rank = 0
        for n_b in n_splits:
            for m_b in m_splits:
                for p_b in p_splits:
                    if n_b == 2 and m_b == 2 and p_b == 2:
                        rank += 7
                    else:
                        rank += n_b * m_b * p_b
        return rank

    def build_decomposition(n_splits, m_splits, p_splits):
        n_offsets = [0]
        for s in n_splits[:-1]:
            n_offsets.append(n_offsets[-1] + s)
        m_offsets = [0]
        for s in m_splits[:-1]:
            m_offsets.append(m_offsets[-1] + s)
        p_offsets = [0]
        for s in p_splits[:-1]:
            p_offsets.append(p_offsets[-1] + s)

        rank = compute_rank(n_splits, m_splits, p_splits)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    n_b = n_splits[i_b]
                    m_b = m_splits[j_b]
                    p_b = p_splits[k_b]
                    i_off = n_offsets[i_b]
                    j_off = m_offsets[j_b]
                    k_off = p_offsets[k_b]

                    if n_b == 2 and m_b == 2 and p_b == 2:
                        for u_entries, v_entries, w_entries in strassen_patterns:
                            for li, lj, val in u_entries:
                                gi = i_off + li
                                gj = j_off + lj
                                U[gi * m + gj][col] = val
                            for lj, lk, val in v_entries:
                                gj = j_off + lj
                                gk = k_off + lk
                                V[gj * p + gk][col] = val
                            for lk, li, val in w_entries:
                                gk = k_off + lk
                                gi = i_off + li
                                W[gk * n + gi][col] = val
                            col += 1
                    else:
                        for li in range(n_b):
                            for lj in range(m_b):
                                for lk in range(p_b):
                                    gi = i_off + li
                                    gj = j_off + lj
                                    gk = k_off + lk
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    col += 1

        return {"U": U, "V": V, "W": W}, rank

    def get_splits(d):
        splits = []
        def _gen(remaining, current):
            if remaining == 0:
                splits.append(current[:])
                return
            for s in range(1, remaining + 1):
                current.append(s)
                _gen(remaining - s, current)
                current.pop()
        _gen(d, [])
        return splits

    n_split_options = get_splits(n)
    m_split_options = get_splits(m)
    p_split_options = get_splits(p)

    all_candidates = []
    for ns in n_split_options:
        for ms in m_split_options:
            for ps in p_split_options:
                r = compute_rank(ns, ms, ps)
                all_candidates.append((r, ns, ms, ps))

    all_candidates.sort(key=lambda x: x[0])

    best_result = None
    best_rank = float('inf')

    for r, ns, ms, ps in all_candidates:
        if time.time() > deadline:
            break
        if r >= best_rank:
            break

        try:
            result, actual_rank = build_decomposition(ns, ms, ps)
            if actual_rank < best_rank:
                best_rank = actual_rank
                best_result = result
        except Exception:
            continue

    # Fallback: try specific known-good splits if search didn't find them
    known_good_splits = [
        ([1, 2], [2, 3], [2, 3]),
        ([2, 1], [2, 3], [2, 3]),
        ([1, 2], [3, 2], [2, 3]),
        ([1, 2], [2, 3], [3, 2]),
        ([2, 1], [3, 2], [2, 3]),
        ([2, 1], [2, 3], [3, 2]),
    ]
    for ns, ms, ps in known_good_splits:
        if time.time() > deadline:
            break
        r = compute_rank(ns, ms, ps)
        if r >= best_rank:
            continue
        try:
            result, actual_rank = build_decomposition(ns, ms, ps)
            if actual_rank < best_rank:
                best_rank = actual_rank
                best_result = result
        except Exception:
            continue

    if best_result is None:
        best_result, _ = build_decomposition([n], [m], [p])

    return best_result
# EVOLVE-BLOCK-END
