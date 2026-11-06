import time
import random
from collections import defaultdict

_START = time.time()
_DEADLINE = _START + 135.0

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_any_cycle(adj, remaining, rng, tries=50):
    """Find a cycle in the subgraph of 'remaining'. Returns list of vertices or None."""
    if not remaining:
        return None
    # Build adjacency restricted to remaining
    adjr = defaultdict(set)
    for u, v in remaining:
        adjr[u].add(v)
        adjr[v].add(u)
    # Try random starts
    verts = list(adjr.keys())
    rng.shuffle(verts)
    for start in verts:
        if len(adjr[start]) < 2:
            continue
        # DFS to find a cycle back to start
        visited = {start}
        path = [start]
        # iterative DFS
        stack = [(start, [start], {start})]
        while stack:
            node, p, vis = stack.pop()
            for nb in list(adjr[node]):
                if nb == start:
                    # found cycle
                    return p + [start]
                if nb not in vis:
                    stack.append((nb, p + [nb], vis | {nb}))
    return None

def _decompose(edges, rng, time_limit):
    remaining = set(edges)
    pieces = []
    while remaining:
        if time.time() > time_limit:
            break
        # Try to find a cycle
        adj = _build_adj(remaining)
        cyc = None
        for _ in range(30):  # a few random restarts
            if time.time() > time_limit:
                break
            c = _find_any_cycle(adj, remaining, rng)
            if c and (cyc is None or len(c) > len(cyc)):
                cyc = c
        if cyc is None:
            break
        # Remove cycle edges
        cyc_edges = set()
        for i in range(len(cyc) - 1):
            u, v = cyc[i], cyc[i + 1]
            cyc_edges.add((min(u, v), max(u, v)))
        remaining -= cyc_edges
        pieces.append(list(cyc))
    for u, v in remaining:
        pieces.append([u, v])
    return pieces

def _build_adj(edge_set):
    adj = defaultdict(set)
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    rng = random.Random(999)
    results = []
    for (n, p, seed) in specs:
        edges = _instance(n, p, seed)
        t_limit = min(_DEADLINE - 5.0, time.time() + 30.0)
        pieces = _decompose(edges, rng, t_limit)
        results.append(pieces)
    return results
