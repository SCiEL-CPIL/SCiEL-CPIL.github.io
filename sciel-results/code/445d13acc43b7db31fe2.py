"""Clique partition of two fixed dense random graphs.
Greedy edge-coverage: repeatedly pick the candidate clique that covers the most
still-unassigned edges; assign each edge to the first chosen clique containing it.
"""
import time

_DEADLINE = 145.0  # safety margin under the 150s wall-clock kill
_T0 = time.time()


def _deadline_hit():
    return (time.time() - _T0) >= _DEADLINE


def _random_graph(n, p, seed):
    from random import Random
    rng = Random(seed)
    adj = [0] * (1 << n) if n <= 20 else None
    # Use bitsets for adjacency when feasible; else set-of-sets.
    adj_set = [set() for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj_set[u].add(v)
                adj_set[v].add(u)
                edges.append((u, v))
    return adj_set, edges


def _expand_clique(seed_clique, adj_set, n):
    """Greedily grow a clique to a maximal one."""
    clique = list(seed_clique)
    cl_set = set(clique)
    # candidates = common neighbours
    if not clique:
        cand = set(range(n))
    else:
        cand = set(adj_set[clique[0]])
        for c in clique[1:]:
            cand &= adj_set[c]
    cand -= cl_set
    while cand:
        # pick the vertex with max degree within cand
        best = max(cand, key=lambda v: len(adj_set[v] & cand))
        clique.append(best)
        cl_set.add(best)
        cand &= adj_set[best]
        cand -= cl_set
    return clique


def _candidate_cliques(adj_set, edges, n):
    """Build a pool of candidate cliques: triangles expanded, plus big greedy ones."""
    cands = []
    seen = set()

    def add(cl):
        if len(cl) < 2:
            return
        key = frozenset(cl)
        if key in seen:
            return
        seen.add(key)
        cands.append(cl)

    # Triangles
    for u in range(n):
        for v in adj_set[u]:
            if v <= u:
                continue
            for w in adj_set[u] & adj_set[v]:
                if w > v:
                    add([u, v, w])

    # Greedy large cliques from each vertex
    for u in range(n):
        add(_expand_clique([u], adj_set, n))

    # Greedy from each edge
    for (u, v) in edges:
        if _deadline_hit():
            break
        add(_expand_clique([u, v], adj_set, n))

    return cands


def _partition(adj_set, edges, n):
    cands = _candidate_cliques(adj_set, edges, n)

    # Map each edge -> list of candidate indices containing it
    edge_to_cands = {e: [] for e in edges}
    for ci, cl in enumerate(cands):
        s = set(cl)
        for u in cl:
            for v in adj_set[u]:
                if v in s and v > u:
                    edge_to_cands[(u, v)].append(ci)

    uncovered = set(edges)
    chosen = []
    # Precompute edge set per candidate for speed
    cand_edges = []
    for cl in cands:
        s = set(cl)
        es = set()
        for u in cl:
            for v in adj_set[u]:
                if v in s and v > u:
                    es.add((u, v))
        cand_edges.append(es)

    while uncovered and not _deadline_hit():
        best_ci = -1
        best_cnt = 0
        for ci, es in enumerate(cand_edges):
            if not es:
                continue
            cnt = len(es & uncovered)
            if cnt > best_cnt:
                best_cnt = cnt
                best_ci = ci
        if best_ci < 0:
            break
        chosen.append(cands[best_ci])
        uncovered -= cand_edges[best_ci]

    # Fallback: any still-uncovered edge becomes its own 2-clique
    for e in list(uncovered):
        if _deadline_hit():
            break
        chosen.append(list(e))
        uncovered.discard(e)

    # Assign each edge to exactly one chosen clique: the first containing it.
    # Build result as list of cliques (vertex lists). Every edge lies in exactly one.
    # Since we only need the partition of edges, return the chosen cliques.
    # But we must ensure no edge is in two returned cliques. Rebuild: for each chosen
    # clique in order, keep only its not-yet-assigned edges; if a clique has no new
    # edges, drop it.
    assigned = set()
    result = []
    for cl in chosen:
        s = set(cl)
        new_edges = []
        for u in cl:
            for v in adj_set[u]:
                if v in s and v > u:
                    e = (u, v)
                    if e not in assigned:
                        new_edges.append(e)
        if new_edges:
            for e in new_edges:
                assigned.add(e)
            result.append(list(cl))

    # Final safety: any edge not assigned (shouldn't happen) -> add 2-clique
    for e in edges:
        if e not in assigned:
            result.append(list(e))
            assigned.add(e)

    return result


def construct():
    out = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        adj_set, edges = _random_graph(n, p, seed)
        out.append(_partition(adj_set, edges, n))
    return out
