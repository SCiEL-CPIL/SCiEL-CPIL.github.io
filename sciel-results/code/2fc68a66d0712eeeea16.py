import time
from random import Random


def _generate_edges(n, p, seed):
    rng = Random(seed)
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return edges


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _solve_instance(n, edges, deadline):
    adj = _build_adj(n, edges)
    uncovered = set(edges)
    cliques = []
    while uncovered:
        if time.time() > deadline:
            break
        # Choose the uncovered edge that can be extended into the largest
        # clique using only currently-uncovered edges.
        best_seed = None
        best_pot = -1
        for (u, v) in uncovered:
            common = adj[u] & adj[v]
            pot = 0
            for w in common:
                a, b = (u, w) if u < w else (w, u)
                c, d = (v, w) if v < w else (w, v)
                if (a, b) in uncovered and (c, d) in uncovered:
                    pot += 1
            if pot > best_pot:
                best_pot = pot
                best_seed = (u, v)
                if pot >= n - 2:
                    break
        if best_seed is None:
            break
        u, v = best_seed
        C = [u, v]
        Cset = {u, v}
        cand = (adj[u] & adj[v]) - Cset
        while cand:
            best_w = None
            best_gain = 0
            for w in cand:
                gain = 0
                for c in C:
                    a, b = (w, c) if w < c else (c, w)
                    if (a, b) in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_w = w
            if best_w is None or best_gain == 0:
                break
            C.append(best_w)
            Cset.add(best_w)
            cand = cand & adj[best_w]
        cliques.append(C)
        for i in range(len(C)):
            a = C[i]
            for j in range(i + 1, len(C)):
                b = C[j]
                uncovered.discard((a, b) if a < b else (b, a))
    for e in uncovered:
        cliques.append(list(e))
    return cliques


def construct():
    start = time.time()
    deadline = start + 115.0
    results = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = _generate_edges(n, p, seed)
        cliques = _solve_instance(n, edges, deadline)
        results.append(cliques)
    return results
