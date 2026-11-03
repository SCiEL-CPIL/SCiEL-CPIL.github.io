"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 150s kill

    # Strassen patterns for 2x2x2 (7 rank-1 terms)
    strassen_patterns = [
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]},
    ]

    def get_splits(d):
        """All compositions of d into parts of size 1 or 2."""
        if d == 0:
            yield []
        elif d == 1:
            yield [1]
        elif d == 2:
            yield [2]
            yield [1, 1]
        else:
            for rest in get_splits(d - 1):
                yield [1] + list(rest)
            for rest in get_splits(d - 2):
                yield [2] + list(rest)

    def compute_rank(n_splits, m_splits, p_splits):
        rank = 0
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                    if nb == 2 and mb == 2 and pb == 2:
                        rank += 7
                    else:
                        rank += nb * mb * pb
        return rank

    def build_decomposition(n_splits, m_splits, p_splits):
        n_off = [0]
        for s in n_splits[:-1]:
            n_off.append(n_off[-1] + s)
        m_off = [0]
        for s in m_splits[:-1]:
            m_off.append(m_off[-1] + s)
        p_off = [0]
        for s in p_splits[:-1]:
            p_off.append(p_off[-1] + s)

        U_entries = []
        V_entries = []
        W_entries = []
        rank = 0

        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                    i_off = n_off[i_b]
                    j_off = m_off[j_b]
                    k_off = p_off[k_b]

                    if nb == 2 and mb == 2 and pb == 2:
                        for pattern in strassen_patterns:
                            col = rank
                            for li, lj, val in pattern['U']:
                                U_entries.append(((i_off + li) * m + (j_off + lj), col, val))
                            for lj, lk, val in pattern['V']:
                                V_entries.append(((j_off + lj) * p + (k_off + lk), col, val))
                            for lk, li, val in pattern['W']:
                                W_entries.append(((k_off + lk) * n + (i_off + li), col, val))
                            rank += 1
                    else:
                        for li in range(nb):
                            for lj in range(mb):
                                for lk in range(pb):
                                    col = rank
                                    U_entries.append(((i_off + li) * m + (j_off + lj), col, 1))
                                    V_entries.append(((j_off + lj) * p + (k_off + lk), col, 1))
                                    W_entries.append(((k_off + lk) * n + (i_off + li), col, 1))
                                    rank += 1

        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        for idx, col, val in U_entries:
            U[idx][col] = val
        for idx, col, val in V_entries:
            V[idx][col] = val
        for idx, col, val in W_entries:
            W[idx][col] = val

        return {"U": U, "V": V, "W": W}

    # Enumerate all splits and find the minimum rank
    best_rank = n * m * p
    best_splits = None
    best_ns, best_ms, best_ps = [n], [m], [p]

    n_opts = list(get_splits(n))
    m_opts = list(get_splits(m))
    p_opts = list(get_splits(p))

    for ns in n_opts:
        for ms in m_opts:
            for ps in p_opts:
                r = compute_rank(ns, ms, ps)
                if r < best_rank:
                    best_rank = r
                    best_ns, best_ms, best_ps = ns, ms, ps
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Build the decomposition for the best split
    result = build_decomposition(best_ns, best_ms, best_ps)
    return result
# EVOLVE-BLOCK-END
