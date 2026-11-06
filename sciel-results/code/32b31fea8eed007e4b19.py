# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START = time.time()
_DEADLINE = 120.0  # wall-clock limit; keep safety margin

def _time_left():
    return _DEADLINE - 5.0 - (time.time() - _START)

def _gen_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
    return adj, edges

def _is_clique(vertices, adj):
    vs = list(vertices)
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            if vs[j] not in adj[vs[i]]:
                return False
    return True

def _greedy_max_clique(adj, n, start=None):
    """Greedy grow a clique starting from a seed vertex."""
    if start is None:
        start = 0
    clique = [start]
    cand = set(adj[start])
    while cand:
        # pick candidate with max degree within cand
        best = max(cand, key=lambda x: len(adj[x] & cand))
        clique.append(best)
        cand = cand & adj[best]
        cand.discard(best)
    return clique

def _solve_instance(adj, edges, n):
    edge_set = set(edges)
    # map edge -> list of vertices; we track uncovered edges
    uncovered = set(edge_set)
    cliques = []
    if not uncovered:
        return cliques

    # Precompute for each vertex the set of incident edges (as frozenset pairs)
    # We'll greedily find cliques that cover many uncovered edges.

    # Strategy: iterate over candidate seed vertices, grow a clique,
    # score it by number of uncovered edges it covers, pick best.
    # To keep it fast, limit candidate seeds and clique sizes.

    # Build a list of candidate seed vertices ordered by degree
    seeds = list(range(n))

    while uncovered and _time_left() > 1:
        best_clique = None
        best_count = 0

        # Try growing cliques from several seeds
        # Use a limited set of seeds: high-degree vertices first
        tried_seeds = 0
        for s in seeds:
            if tried_seeds >= 40:
                break
            tried_seeds += 1
            cl = _greedy_max_clique(adj, n, start=s)
            if len(cl) < 2:
                continue
            # Count uncovered edges in this clique
            count = 0
            cl_set = set(cl)
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    e = (cl[i], cl[j])
                    if e in uncovered:
                        count += 1
            if count > best_count:
                best_count = count
                best_clique = cl
            if best_count == 0:
                break

        if best_clique is None or best_count == 0:
            # Fallback: cover remaining edges one by one
            for e in list(uncovered):
                cliques.append(list(e))
                uncovered.discard(e)
            break

        cliques.append(best_clique)
        cl_set = set(best_clique)
        # Remove covered edges
        for i in range(len(best_clique)):
            for j in range(i + 1, len(best_clique)):
                e = (best_clique[i], best_clique[j])
                uncovered.discard(e)

    # If any edges remain uncovered (shouldn't happen but safety)
    for e in uncovered:
        cliques.append(list(e))

    return cliques

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        adj, edges = _gen_graph(n, p, seed)
        cliques = _solve_instance(adj, edges, n)
        results.append(cliques)
    return results
# EVOLVE-BLOCK-END
