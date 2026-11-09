"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _alpha_bnb(adj, n):
    best = [0]
    def bnb(cand, sz):
        if sz + cand.bit_count() <= best[0]:
            return
        if cand == 0:
            if sz > best[0]:
                best[0] = sz
            return
        best_v = -1
        best_deg = -1
        for v in range(n):
            if (cand >> v) & 1:
                deg = (adj[v] & cand).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        v_bit = 1 << best_v
        bnb(cand & ~v_bit & ~adj[best_v], sz + 1)
        bnb(cand & ~v_bit, sz)
    bnb((1 << n) - 1, 0)
    return best[0]

def _try_cayley(S, n):
    S_set = set(S)
    for s1 in S_set:
        for s2 in S_set:
            if (s1 + s2) % n in S_set:
                return None
    adj = [0] * n
    for s in S_set:
        neg_s = (-s) % n
        for i in range(n):
            j1 = (i + s) % n
            j2 = (i + neg_s) % n
            if j1 != i:
                adj[i] |= (1 << j1)
            if j2 != i:
                adj[i] |= (1 << j2)
    alpha = _alpha_bnb(adj, n)
    if alpha < 10:
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    candidates = [
        (39, set(s for s in range(1, 39) if s % 5 in (1, 2))),
        (40, set(s for s in range(1, 40) if s % 5 in (1, 2))),
        (39, set(s for s in range(1, 39) if s % 7 in (1, 2, 3))),
        (40, set(s for s in range(1, 40) if s % 7 in (1, 2, 3))),
        (39, set(s for s in range(1, 39) if s % 4 in (1, 2))),
        (40, set(s for s in range(1, 40) if s % 4 in (1, 2))),
        (39, set(s for s in range(1, 39) if s % 6 in (1, 2))),
        (40, set(s for s in range(1, 40) if s % 6 in (1, 2))),
    ]
    for n, S in candidates:
        if time.time() > deadline:
            break
        result = _try_cayley(S, n)
        if result:
            return result
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
# EVOLVE-BLOCK-END
