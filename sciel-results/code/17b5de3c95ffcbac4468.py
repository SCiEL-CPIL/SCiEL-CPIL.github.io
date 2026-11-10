"""Rewritten program: weighted greedy set-cover over maximal cliques (size>=2)."""
# EVOLVE-BLOCK-START
import time

DEADLINE = 150.0  # wall-clock limit for evaluator; we keep a safety margin
SAFETY_MARGIN = 5.0  # seconds to leave before the hard kill


def _build_graph(n, p, seed):
    """Generate the same random graph as the seed program: edges (u,v) with u<v."""
    import random
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def _bron_kerbosch(adj):
    """Enumerate all maximal cliques (as frozensets) using Bron-Kerbosch with pivot."""
    n = len(adj)
    maximal_cliques = []

    def expand(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                maximal_cliques.append(frozenset(R))
            return
        # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
        union = P | X
        if not union:
            return
        # Pick pivot with max overlap with P
        best_u = None
        best_count = -1
        for u in union:
            cnt = len(P & adj[u])
            if cnt > best_count:
                best_count = cnt
                best_u = u
        # Iterate over P \ N(best_u)
        candidates = [v for v in P if best_u not in adj[v]]
        for v in candidates:
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            P.discard(v)
            expand(new_R, new_P, new_X)
            X.add(v)

    expand(frozenset(), set(range(n)), set())
    return maximal_cliques


def _greedy_set_cover(cliques, n):
    """
    Greedy weighted set cover: repeatedly pick the vertex that covers the most
    uncovered cliques, weighted by 1/|clique| (harder cliques get higher weight).
    """
    if not cliques:
        return []

    # Weight each clique by 1/len(clique) — smaller cliques are "harder" to cover
    clique_weights = {i: 1.0 / len(c) for i, c in enumerate(cliques)}

    # For each vertex, which cliques it belongs to
    vertex_to_cliques = {v: set() for v in range(n)}
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_to_cliques[v].add(i)

    uncovered = set(range(len(cliques)))
    selected = []

    while uncovered:
        best_v = None
        best_score = -1
        for v in range(n):
            if v in selected:
                continue
            # Score = sum of weights of uncovered cliques that v covers
            score = sum(clique_weights[i] for i in vertex_to_cliques[v] if i in uncovered)
            if score > best_score:
                best_score = score
                best_v = v
        if best_v is None or best_score <= 0:
            break
        selected.append(best_v)
        # Remove all cliques covered by best_v
        covered = vertex_to_cliques[best_v] & uncovered
        uncovered -= covered

    return sorted(selected)


def _solve_instance(n, p, seed, time_budget):
    """Solve one instance within the given time budget."""
    start = time.time()
    adj = _build_graph(n, p, seed)
    cliques = _bron_kerbosch(adj)
    elapsed = time.time() - start
    remaining = time_budget - elapsed
    if remaining < 1.0:
        # Not enough time; return a trivial valid solution (all vertices)
        return list(range(n))
    result = _greedy_set_cover(cliques, n)
    return result


def construct():
    """
    Returns three vertex lists, one per instance from random_graph(n, p, seed)
    for (30,0.3,201), (35,0.25,202), (40,0.2,203).
    Each list meets every maximal clique with at least two vertices.
    Minimize total size.
    """
    start_time = time.time()
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []

    for i, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        # Allocate remaining time divided among remaining instances, with safety margin
        remaining_instances = len(instances) - i
        time_budget = max(1.0, (DEADLINE - SAFETY_MARGIN - elapsed) / remaining_instances)

        result = _solve_instance(n, p, seed, time_budget)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
