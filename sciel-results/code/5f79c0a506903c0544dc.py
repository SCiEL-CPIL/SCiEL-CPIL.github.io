import time
import random

_N = 18
_ALL = (1 << _N) - 1
_DEADLINE = 130.0  # safety margin before the 150 s wall-clock kill

# Precomputed popcount table for 18-bit masks (fast upper bounds / picks)
_PC = [0] * (1 << _N)
for _i in range(1, 1 << _N):
    _PC[_i] = _PC[_i >> 1] + (_i & 1)


def _pc(x):
    return _PC[x]


def _safe(w, S_mask, adj):
    """True iff adding w to triangle-free set S keeps it triangle-free,
    i.e. the neighbours of w inside S form an independent set."""
    N = adj[w] & S_mask
    n = N
    while n:
        lsb = n & -n
        x = lsb.bit_length() - 1
        n ^= lsb
        if adj[x] & N:
            return False
    return True


def _max_tf_size(adj):
    """Exact largest triangle-free induced subgraph size via branch & bound."""

    def dfs(S_mask, cand_mask, best_ref):
        ub = _PC[S_mask] + _PC[cand_mask]
        if ub <= best_ref[0]:
            return
        if cand_mask == 0:
            s = _PC[S_mask]
            if s > best_ref[0]:
                best_ref[0] = s
            return
        # pick v in cand with highest degree inside cand (better pruning)
        c = cand_mask
        best_v = -1
        best_d = -1
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            d = _PC[adj[v] & cand_mask]
            if d > best_d:
                best_d = d
                best_v = v
        v = best_v
        # branch: include v
        newS = S_mask | (1 << v)
        newcand = 0
        cc = cand_mask & ~(1 << v)
        while cc:
            lsb = cc & -c
            w = lsb.bit_length() - 1
            cc ^= lsb
            if _safe(w, newS, adj):
                newcand |= 1 << w
        dfs(newS, newcand, best_ref)
        # branch: exclude v
        dfs(S_mask, cand_mask & ~(1 << v), best_ref)

    # greedy lower bound to seed the bound
    greedy_S = 0
    cand = _ALL
    while cand:
        lsb = cand & -cand
        v = lsb.bit_length() - 1
        cand ^= lsb
        if _safe(v, greedy_S, adj):
            greedy_S |= 1 << v
    best_ref = [_PC[greedy_S]]
    dfs(0, _ALL, best_ref)
    return best_ref[0]


def _random_maximal_k4free(rng, adj0=None):
    """Insert edges in random order, accepting an edge only if it creates no K4.
    Result is a maximal K4-free graph."""
    adj = [0] * _N if adj0 is None else list(adj0)
    edges = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
    rng.shuffle(edges)
    for (u, v) in edges:
        if (adj[u] >> v) & 1:
            continue
        common = adj[u] & adj[v]
        ok = True
        c = common
        while c:
            lsb = c & -c
            w = lsb.bit_length() - 1
            c ^= lsb
            if adj[w] & common:  # w-x edge inside common -> K4 {u,v,w,x}
                ok = False
                break
        if ok:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
    return adj


def _is_k4_free(adj):
    for a in range(_N):
        for b in range(a + 1, _N):
            if not ((adj[a] >> b) & 1):
                continue
            for c in range(b + 1, _N):
                if not (((adj[a] >> c) & 1) and ((adj[b] >> c) & 1)):
                    continue
                for d in range(c + 1, _N):
                    if ((adj[a] >> d) & 1) and ((adj[b] >> d) & 1) and ((adj[c] >> d) & 1):
                        return False
    return True


def _adj_to_edges(adj):
    out = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (adj[u] >> v) & 1:
                out.append([u, v])
    return out


def construct():
    start = time.time()
    rng = random.Random(20240517)

    best_adj = None
    best_score = _N + 1

    def consider(adj):
        nonlocal best_adj, best_score
        if not _is_k4_free(adj):
            return
        s = _max_tf_size(adj)
        if s < best_score:
            best_score = s
            best_adj = list(adj)

    # seed the search with the tripartite construction (known K4-free, score 12)
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    seed_adj = [0] * _N
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    seed_adj[a] |= 1 << b
                    seed_adj[b] |= 1 << a
    consider(seed_adj)

    iters = 0
    while time.time() - start < _DEADLINE:
        # primary: fresh random maximal K4-free graph
        adj = _random_maximal_k4free(rng)
        consider(adj)
        iters += 1

        # local-search perturbation: drop one edge from the current best,
        # then re-maximize (different shuffle) to explore neighbours
        if best_adj is not None and rng.random() < 0.35:
            a2 = list(best_adj)
            es = []
            for u in range(_N):
                for v in range(u + 1, _N):
                    if (a2[u] >> v) & 1:
                        es.append((u, v))
            if es:
                u, v = rng.choice(es)
                a2[u] &= ~(1 << v)
                a2[v] &= ~(1 << u)
                a3 = _random_maximal_k4free(rng, a2)
                consider(a3)

        # if we already hit a very strong bound, keep refining but stop early
        if best_score <= 9:
            # 9 is excellent; do a short burst of extra perturbations then exit
            for _ in range(200):
                if time.time() - start > _DEADLINE - 5:
                    break
                if best_adj is None:
                    break
                a2 = list(best_adj)
                es = [(u, v) for u in range(_N) for v in range(u + 1, _N)
                      if (a2[u] >> v) & 1]
                if not es:
                    break
                u, v = rng.choice(es)
                a2[u] &= ~(1 << v)
                a2[v] &= ~(1 << u)
                consider(_random_maximal_k4free(rng, a2))
            break

    if best_adj is None:
        return [[a, b] for i in range(3) for j in range(i + 1, 3)
                for a in parts[i] for b in parts[j]]
    return _adj_to_edges(best_adj)
