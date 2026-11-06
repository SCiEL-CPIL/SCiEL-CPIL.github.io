"""Q3-free graph on 16 vertices: maximize edges via exact bitmask cube detection."""
import time
import random
import itertools

_N = 16

_ALL8V = []
for _combo in itertools.combinations(range(_N), 8):
    _m = 0
    for _v in _combo:
        _m |= 1 << _v
    _ALL8V.append((_m, _combo))

_SPLITS = []
for _a in itertools.combinations(range(8), 4):
    _am = 0
    for _i in _a:
        _am |= 1 << _i
    _SPLITS.append((_am, 0xFF ^ _am))

_ALL_PAIRS = [(u, v) for u in range(_N) for v in range(u + 1, _N)]

_PAIR_SETS = {}
for (_u, _v) in _ALL_PAIRS:
    _uv = (1 << _u) | (1 << _v)
    _PAIR_SETS[(_u, _v)] = [(m, verts) for (m, verts) in _ALL8V if (m & _uv) == _uv]


def _has_q3_with_edge(adj, u, v):
    for (m, verts) in _PAIR_SETS[(u, v)]:
        ok = True
        for i in verts:
            if (adj[i] & m).bit_count() < 3:
                ok = False
                break
        if not ok:
            continue
        nm = [adj[i] & m for i in verts]
        for am, bm in _SPLITS:
            good = True
            for a in range(8):
                if (am >> a) & 1:
                    if (nm[a] & bm).bit_count() != 3:
                        good = False
                        break
            if not good:
                continue
            for b in range(8):
                if (bm >> b) & 1:
                    if (nm[b] & am).bit_count() != 3:
                        good = False
                        break
            if good:
                return True
    return False


def _try_add(adj, edges, u, v):
    if (adj[u] >> v) & 1:
        return True
    adj[u] |= 1 << v
    adj[v] |= 1 << u
    if _has_q3_with_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return False
    edges.add((u, v))
    return True


def _greedy(order, deadline):
    adj = [0] * _N
    edges = set()
    for (u, v) in order:
        if (time.time() - deadline[0]) > deadline[1]:
            break
        if (adj[u] >> v) & 1:
            continue
        _try_add(adj, edges, u, v)
    changed = True
    while changed and (time.time() - deadline[0]) < deadline[1]:
        changed = False
        for (u, v) in _ALL_PAIRS:
            if (adj[u] >> v) & 1:
                continue
            if _try_add(adj, edges, u, v):
                changed = True
    return adj, edges


def _local_grow(adj, edges, deadline, rng):
    best_adj = adj[:]
    best_edges = set(edges)
    best_n = len(edges)
    while (time.time() - deadline[0]) < deadline[1]:
        if not edges:
            break
        e = rng.choice(list(edges))
        u, v = e
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edges.discard(e)
        added = 0
        candidates = [p for p in _ALL_PAIRS if p not in edges and (adj[p[0]] >> p[1]) == 0]
        rng.shuffle(candidates)
        for (a, b) in candidates:
            if added >= 2:
                break
            if _try_add(adj, edges, a, b):
                added += 1
        if len(edges) > best_n:
            best_n = len(edges)
            best_adj = adj[:]
            best_edges = set(edges)
            adj = best_adj[:]
            edges = set(best_edges)
        else:
            adj = best_adj[:]
            edges = set(best_edges)
    return best_adj, best_edges


def construct():
    t0 = time.time()
    DEADLINE = 110.0
    deadline = (t0, DEADLINE)

    best_adj = [0] * _N
    best_edges = set()
    best_n = 0

    adj, edges = _greedy(_ALL_PAIRS, deadline)
    if len(edges) > best_n:
        best_n = len(edges)
        best_adj = adj[:]
        best_edges = set(edges)

    rng = random.Random(20240517)
    while (time.time() - t0) < DEADLINE - 5.0:
        order = _ALL_PAIRS[:]
        rng.shuffle(order)
        adj, edges = _greedy(order, deadline)
        if len(edges) > best_n:
            best_n = len(edges)
            best_adj = adj[:]
            best_edges = set(edges)
        if (time.time() - t0) < DEADLINE - 5.0:
            gadj, gedges = _local_grow(best_adj[:], set(best_edges), deadline, rng)
            if len(gedges) > best_n:
                best_n = len(gedges)
                best_adj = gadj[:]
                best_edges = set(gedges)

    return [[u, v] for (u, v) in best_edges]
