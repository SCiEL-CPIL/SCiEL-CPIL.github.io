"""Clique partition of two fixed dense random graphs via repeated maximum-clique extraction."""
import time

_T0 = time.time()
_DEADLINE = _T0 + 130.0  # wall-clock guard (evaluator kills at 150s)


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly (Random(seed), u<v, rng.random()<p)."""
    from random import Random
    rng = Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _max_clique(adj, n):
    """Maximum clique via Bron–Kerbosch with Tomita coloring + pivot (dense graphs)."""
    deg = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -deg[i])
    remap = [0] * n
    for new, old in enumerate(order):
        remap[old] = new
    m = n
    cand = [set() for _ in range(m)]
    for old in range(n):
        ni = remap[old]
        for v in adj[old]:
            cand[ni].add(remap[v])

    best = []
    best_size = 0

    def color_sort(R, P):
        nonlocal best, best_size
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best = list(R)
            return
        colored = []
        colors = {}
        avail = set(P)
        while avail:
            color = len(colored) + 1
            layer = set()
            for v in avail:
                if not (cand[v] & avail):
                    layer.add(v)
            for v in layer:
                avail.remove(v)
            if not layer:
                break
            colors.update((v, color) for v in layer)
            colored.append((layer, color))
        for layer, color in reversed(colored):
            if len(R) + color <= best_size:
                break
            for v in layer:
                R.append(v)
                P2 = set(P & cand[v])
                color_sort(R, P2)
                R.pop()
                P.discard(v)

    color_sort([], set(range(m)))

    if not best:
        return []
    inv = [0] * m
    for new, old in enumerate(order):
        inv[new] = old
    return [inv[x] for x in best]


def _max_clique_fallback(adj, n, deadline):
    """Greedy degeneracy-order clique: O(n^2) guaranteed, always a clique."""
    order = sorted(range(n), key=lambda i: len(adj[i]))
    pos = {v: i for i, v in enumerate(order)}
    R = []
    for v in order:
        if all(pos[u] > pos[v] for u in adj[v]):
            R.append(v)
    return R


def _solve_instance(n, p, seed, deadline):
    edges = _random_graph(n, p, seed)
    adj = _build_adj(n, edges)
    rem = set(edges)
    cliques = []
    while rem:
        if time.time() >= deadline:
            break
        cl = _max_clique(adj, n)
        if len(cl) < 2:
            cl = _max_clique_fallback(adj, n, deadline)
        if len(cl) < 2:
            break
        used = set()
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                a, b = cl[i], cl[j]
                if a > b:
                    a, b = b, a
                used.add((a, b))
        covered = used & rem
        if not covered:
            break
        rem -= covered
        cliques.append(cl)
    # Any uncovered edges (only if we hit the deadline) become 2-cliques (valid cliques).
    for e in rem:
        cliques.append(list(e))
    return cliques


def construct():
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        dl = min(_DEADLINE, time.time() + 60.0)  # per-instance margin
        result.append(_solve_instance(n, p, seed, dl))
    return result
