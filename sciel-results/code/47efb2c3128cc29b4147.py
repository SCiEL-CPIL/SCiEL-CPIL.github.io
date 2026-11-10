"""Rewritten: structural edge-cover minimization via 3-clique merging (line-graph matching)."""
import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin under 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        m = len(edges)
        edge_set = set(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Line graph: edges i,j adjacent iff they share a vertex.
        # Build adjacency lists by vertex-incident edges.
        by_vertex = [set() for _ in range(n)]
        for idx, (u, v) in enumerate(edges):
            by_vertex[u].add(idx)
            by_vertex[v].add(idx)
        lg_adj = [set() for _ in range(m)]
        for u in range(n):
            inc = list(by_vertex[u])
            for a in inc:
                for b in inc:
                    if a < b:
                        lg_adj[a].add(b)
                        lg_adj[b].add(a)

        # Greedy maximal matching on line graph (deterministic order).
        matched = [False] * m
        pairs = []
        for i in range(m):
            if matched[i]:
                continue
            for j in sorted(lg_adj[i]):
                if matched[j]:
                    continue
                (u1, v1) = edges[i]
                (u2, v2) = edges[j]
                shared = set((u1, v1)) & set((u2, v2))
                if len(shared) != 1:
                    continue
                s = next(iter(shared))
                a = u1 if u1 != s else v1
                b = u2 if u2 != s else v2
                if a == b:
                    continue
                if (min(a, b), max(a, b)) in edge_set:
                    matched[i] = True
                    matched[j] = True
                    pairs.append((i, j, (s, a, b)))
                    break

        # Build partition: merged 3-cliques + singleton edges for unmatched.
        cliques = []
        for (i, j, tri) in pairs:
            cliques.append(list(tri))
        for idx in range(m):
            if not matched[idx]:
                cliques.append(list(edges[idx]))
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            break
        results.append(solve(*spec))

    # Ensure exactly three lists; pad if deadline hit early (shouldn't happen).
    while len(results) < 3:
        results.append([])

    # Convert to pure Python lists of ints (guard against numpy types).
    out = []
    for lst in results:
        cl = []
        for c in lst:
            cl.append([int(x) for x in c])
        out.append(cl)
    return out

# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
