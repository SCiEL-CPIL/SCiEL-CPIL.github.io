"""Clique partition of two fixed dense random graphs.

Design (fresh): treat the problem as partitioning the edge set into cliques.
Build the graph, then repeatedly pick a clique (set of mutually adjacent edges
in the line graph, i.e. edges sharing a common vertex) that covers the most
still-uncovered edges, commit it, and remove those edges. This is a greedy
set-cover heuristic on the line graph. A deadline guard returns the best
valid partition found before the wall-clock limit.
"""
import time

_DEADLINE = 150.0
_MARGIN = 5.0


def _generate(n, p, seed):
    """Reproduce random_graph(n, p, seed) exactly as the seed program does."""
    from random import Random
    rng = Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    return edges


def _build_adjacency(edges):
    """Map each edge to a set of other edges that share a vertex (line graph)."""
    m = len(edges)
    # vertex -> list of edge indices
    v2e = {}
    for i, (u, v) in enumerate(edges):
        v2e.setdefault(u, []).append(i)
        v2e.setdefault(v, []).append(i)
    # adjacency: edge i is adjacent to edge j if they share a vertex
    adj = [set() for _ in range(m)]
    for vs in v2e.values():
        L = len(vs)
        for a in range(L):
            ia = vs[a]
            for b in range(a + 1, L):
                ib = vs[b]
                adj[ia].add(ib)
                adj[ib].add(ia)
    return adj


def _greedy_partition(n, edges, adj, deadline):
    """Greedily cover all edges with cliques (sets of pairwise adjacent edges).

    A clique in the line graph = a set of edges that all share a common vertex
    (a star) OR a triangle. We find maximal cliques among uncovered edges.
    For each candidate center vertex, the uncovered edges incident to it form
    a star (a clique). We also consider triangles. We pick the candidate that
    covers the most uncovered edges.
    """
    # vertex -> list of edge indices
    v2e = {}
    for i, (u, v) in enumerate(edges):
        v2e.setdefault(u, []).append(i)
        v2e.setdefault(v, []).append(i)

    # edge -> set of vertices it touches
    e2v = [set() for _ in range(len(edges))]
    for i, (u, v) in enumerate(edges):
        e2v[i] = {u, v}

    uncovered = set(range(len(edges)))
    cliques = []

    def try_commit(cand):
        # cand: list of edge indices that form a clique
        if len(cand) < 2:
            return False
        # verify all pairs adjacent
        for a in range(len(cand)):
            for b in range(a + 1, len(cand)):
                if cand[b] not in adj[cand[a]]:
                    return False
        for e in cand:
            uncovered.discard(e)
        cliques.append([list(edges[e]) for e in cand])
        return True

    # Precompute for each vertex the full incident edge list
    inc = {v: list(es) for v, es in v2e.items()}

    while uncovered and time.time() < deadline - _MARGIN:
        best_cand = None
        best_size = 1
        # candidate stars: for each vertex, uncovered incident edges form a clique
        for v in range(n):
            lst = inc[v]
            if not lst:
                continue
            cand = [e for e in lst if e in uncovered]
            if len(cand) > best_size:
                best_size = len(cand)
                best_cand = cand
        if best_cand is None or best_size < 2:
            # remaining edges are isolated (no two share a vertex) -> each is its own
            # but every edge must be in a clique of >=2 vertices, and an edge alone
            # is a 2-vertex clique. So commit single edges as 2-vertex cliques.
            for e in list(uncovered):
                cliques.append(list(edges[e]))
                uncovered.discard(e)
            break
        try_commit(best_cand)

    return cliques


def construct():
    start = time.time()
    deadline = start + _DEADLINE
    results = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = _generate(n, p, seed)
        adj = _build_adjacency(edges)
        cliques = _greedy_partition(n, edges, adj, deadline)
        results.append(cliques)
        if time.time() > deadline - _MARGIN:
            # not enough time for the second instance; fall back to trivial
            # edge-as-clique partition for remaining instances to stay valid.
            break
    # If we ran out of time, ensure we still return two valid partitions.
    while len(results) < 2:
        n, p, seed = ((32, 0.6, 601), (64, 0.55, 602))[len(results)]
        edges = _generate(n, p, seed)
        results.append([list(e) for e in edges])
    return results
