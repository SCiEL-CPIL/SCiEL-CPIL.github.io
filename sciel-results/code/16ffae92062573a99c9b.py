"""Improved program: greedy simple-path decomposition minimizing path count."""
# EVOLVE-BLOCK-START
from random import Random
from collections import defaultdict

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_graph(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _decompose(edges):
    """Greedy: repeatedly take an unused edge and extend it maximally
    into a simple path (no repeated vertices), using only unused edges."""
    adj = _build_graph(edges)
    used = set()
    for u, v in edges:
        used.add((u, v))
    paths = []

    # Order edges to prefer high-degree endpoints first (heuristic).
    deg = {v: len(adj[v]) for v in adj}
    remaining = list(used)

    while remaining:
        # pick an edge with a high-degree endpoint to encourage long paths
        remaining.sort(key=lambda e: (deg[e[0]] + deg[e[1]]), reverse=True)
        u0, v0 = remaining.pop(0)

        # Build path starting as [u0, v0], extend both ends.
        path = [u0, v0]
        used.remove((u0, v0))

        # Extend from v0 side
        cur = v0
        visited = {u0, v0}
        while True:
            # choose neighbor not visited, with edge still unused
            cands = [w for w in adj[cur] if w not in visited and (cur, w) not in used and (w, cur) not in used]
            if not cands:
                break
            # prefer neighbor with more unused edges (degree) to keep extending
            cands.sort(key=lambda w: deg[w], reverse=True)
            nxt = cands[0]
            e = (cur, nxt) if (cur, nxt) in used else (nxt, cur)
            used.remove(e)
            path.append(nxt)
            visited.add(nxt)
            cur = nxt

        # Extend from u0 side (prepend)
        cur = u0
        while True:
            cands = [w for w in adj[cur] if w not in visited and (cur, w) not in used and (w, cur) not in used]
            if not cands:
                break
            cands.sort(key=lambda w: deg[w], reverse=True)
            nxt = cands[0]
            e = (cur, nxt) if (cur, nxt) in used else (nxt, cur)
            used.remove(e)
            path.insert(0, nxt)
            visited.add(nxt)
            cur = nxt

        paths.append(path)
        # rebuild remaining
        remaining = list(used)

    return paths

def construct():
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    result = []
    for spec in specs:
        edges = instance(*spec)
        paths = _decompose(edges)
        # convert to plain lists of ints
        result.append([list(p) for p in paths])
    return result
# EVOLVE-BLOCK-END
