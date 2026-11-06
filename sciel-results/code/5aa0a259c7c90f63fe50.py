"""Clique partitioning of two dense random graphs: greedy maximal-clique packing + local improvement."""
# EVOLVE-BLOCK-START
import time
from random import Random
from itertools import combinations

_DEADLINE = 120.0
_MARGIN = 8.0


def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _is_clique(vertices, adj):
    vs = list(vertices)
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            if vs[j] not in adj[vs[i]]:
                return False
    return True


def _greedy_maximal_clique(start, adj, remaining):
    """Greedy maximal clique: repeatedly add the vertex with max common neighbors."""
    clique = [start]
    cand = set(adj[start]) & remaining
    while cand:
        # choose candidate maximizing |cand ∩ adj[v]|
        best = None
        best_score = -1
        for v in cand:
            s = len(cand & adj[v])
            if s > best_score:
                best_score = s
                best = v
        if best is None or best_score == 0:
            break
        clique.append(best)
        cand = {x for x in cand if x != best and x in adj[best]}
    return clique


def _pack_cliques(edges, adj, deadline_fn):
    """Greedy: while uncovered edges remain, find a maximal clique in the uncovered subgraph."""
    covered = set()
    cliques = []
    # order edges by a heuristic: prefer edges in denser neighborhoods
    edge_list = list(edges)
    while edge_list:
        if deadline_fn() < 1.0:
            break
        # pick an uncovered edge; use the one with most common neighbors (densest)
        best_idx = 0
        best_common = -1
        # sample for speed on large graphs
        if len(edge_list) > 400:
            sample = edge_list[:400]
        else:
            sample = edge_list
        for i, e in enumerate(sample):
            u, v = e
            c = len(adj[u] & adj[v])
            if c > best_common:
                best_common = c
                best_idx = i
        e = edge_list.pop(best_idx)
        u, v = e
        # build maximal clique containing u and v within uncovered edges
        cand = (adj[u] & adj[v])
        # restrict to vertices that still have an uncovered edge
        clique = [u, v]
        remaining_cand = set(cand)
        while remaining_cand:
            best = None
            best_score = -1
            for x in remaining_cand:
                # x must be adjacent to all current clique members
                ok = all(x in adj[m] for m in clique)
                if not ok:
                    continue
                s = len(remaining_cand & adj[x])
                if s > best_score:
                    best_score = s
                    best = x
            if best is None or best_score == 0:
                break
            clique.append(best)
            remaining_cand = {x for x in remaining_cand if x != best and x in adj[best]}
        # only accept if clique has >=2 and is a valid clique
        if len(clique) >= 2 and _is_clique(clique, adj):
            cliques.append(clique)
            for a, b in combinations(clique, 2):
                covered.add((min(a, b), max(a, b)))
        else:
            cliques.append([u, v])
            covered.add((min(u, v), max(u, v)))
    # add any leftover uncovered edges as 2-cliques
    for e in edge_list:
        cliques.append(list(e))
        covered.add((min(e[0], e[1]), max(e[0], e[1])))
    return cliques


def _local_improve(cliques, adj, deadline_fn):
    """Merge cliques that union to a clique; split/replace to reduce count."""
    changed = True
    while changed and deadline_fn() > 0.5:
        changed = False
        i = 0
        n = len(cliques)
        while i < n:
            if deadline_fn() < 0.5:
                break
            j = i + 1
            while j < n:
                if deadline_fn() < 0.5:
                    break
                a = cliques[i]
                b = cliques[j]
                merged = list(set(a) | set(b))
                if len(merged) >= 2 and _is_clique(merged, adj):
                    # replace both with merged (only if it reduces count, i.e. always 2->1)
                    cliques[i] = merged
                    del cliques[j]
                    n = len(cliques)
                    changed = True
                    break
                j += 1
            if changed:
                break
            i += 1
    return cliques


def _solve_instance(n, p, seed):
    rng = Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    adj = _build_adj(n, edges)
    cliques = _pack_cliques(edges, adj, _remaining)
    cliques = _local_improve(cliques, adj, _remaining)
    return cliques


def construct():
    global _START
    _START = time.time()
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() < 2.0:
            # not enough time; fall back to trivial edge partition
            rng = Random(seed)
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
            results.append([list(e) for e in edges])
            continue
        results.append(_solve_instance(n, p, seed))
    return results


_START = time.time()
# EVOLVE-BLOCK-END
