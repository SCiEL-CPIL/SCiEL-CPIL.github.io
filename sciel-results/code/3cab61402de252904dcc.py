"""Fresh design: local search over edge sets to minimize max triangle degree sum."""
import time
import random

_START = None

def _init():
    global _START
    if _START is None:
        _START = time.time()

def _elapsed():
    return time.time() - _START

def _adj_and_deg(adj, n):
    deg = [0] * n
    for u in range(n):
        deg[u] = adj[u].__len__() if hasattr(adj[u], '__len__') else 0
    # recompute properly
    deg = [0] * n
    for u in range(n):
        for v in adj[u]:
            deg[u] += 1
    return deg

def _max_tri_sum(adj, deg, n):
    best = -1
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            # common neighbors w > v to avoid dup? we just need max, dup ok but costly
            common = adj[u] & adj[v]
            for w in common:
                s = deg[u] + deg[v] + deg[w]
                if s > best:
                    best = s
    return best

def _score(adj, n):
    deg = _adj_and_deg(adj, n)
    return _max_tri_sum(adj, deg, n)

def _seed_graph(n):
    # start from K_{8,8} + one intra edge (the known 25-ish baseline), then perturb
    adj = [set() for _ in range(n)]
    half = n // 2
    for u in range(half):
        for v in range(half, n):
            adj[u].add(v)
            adj[v].add(u)
    adj[0].add(1)
    adj[1].add(0)
    return adj

def _edge_count(adj, n):
    c = 0
    for u in range(n):
        for v in adj[u]:
            if v > u:
                c += 1
    return c

def _add_edge(adj, u, v):
    adj[u].add(v)
    adj[v].add(u)

def _remove_edge(adj, u, v):
    adj[u].discard(v)
    adj[v].discard(u)

def construct():
    _init()
    DEADLINE = 140.0  # safety margin under 150s wall clock
    n = 16
    target = 65

    random.seed(12345)

    def to_edges(adj):
        out = []
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    out.append([u, v])
        return out

    # Build initial candidate: K_{8,8}+edge has 65 edges. Good start.
    adj = _seed_graph(n)
    assert _edge_count(adj, n) == target, _edge_count(adj, n)
    best_adj = [set(s) for s in adj]
    best_score = _score(best_adj, n)

    # Local search: random edge flips that keep edge count fixed.
    # Move: remove one existing edge, add one missing edge (swap), preserving count.
    # Objective: minimize max triangle degree sum.

    def neighbors_of(u):
        return adj[u]

    def try_swap(u1, v1, u2, v2):
        # remove (u1,v1), add (u2,v2); must be distinct edges, (u2,v2) not present, (u1,v1) present
        if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
            return False
        if v1 in adj[u1]:
            pass
        else:
            return False
        if u2 in adj[v2]:
            return False
        _remove_edge(adj, u1, v1)
        _add_edge(adj, u2, v2)
        sc = _score(adj, n)
        if sc < best_score:
            best_score = sc
            best_adj = [set(s) for s in adj]
            return True
        else:
            # revert
            _remove_edge(adj, u2, v2)
            _add_edge(adj, u1, v1)
            return False

    # Precompute list of edges and non-edges for sampling
    def get_edges_list():
        L = []
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    L.append((u, v))
        return L

    def get_nonedges_list():
        L = []
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    L.append((u, v))
        return L

    iters = 0
    while _elapsed() < DEADLINE:
        iters += 1
        # Occasionally do a random restart with perturbation to escape local optima
        if iters % 5000 == 0 and random.random() < 0.3:
            # perturb: do a few random swaps regardless of score
            for _ in range(3):
                EL = get_edges_list()
                NL = get_nonedges_list()
                if not EL or not NL:
                    break
                e1 = random.choice(EL)
                e2 = random.choice(NL)
                _remove_edge(adj, e1[0], e1[1])
                _add_edge(adj, e2[0], e2[1])
            # recompute best if improved
            sc = _score(adj, n)
            if sc < best_score:
                best_score = sc
                best_adj = [set(s) for s in adj]

        EL = get_edges_list()
        NL = get_nonedges_list()
        if not EL or not NL:
            break
        e1 = random.choice(EL)
        e2 = random.choice(NL)
        try_swap(e1[0], e1[1], e2[0], e2[1])

        # also try single-edge "move": remove edge and add edge sharing a vertex? keep simple.

    return {'n': n, 'edges': to_edges(best_adj)}
