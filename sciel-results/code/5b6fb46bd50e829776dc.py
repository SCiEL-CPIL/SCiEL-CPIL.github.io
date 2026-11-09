"""Clique partition of dense random graphs: minimize number of cliques covering every edge."""
import time
from random import Random


def construct():
    _t0 = time.time()
    _DEADLINE = _t0 + 110.0  # safety margin before 120s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def partition(edges, n, deadline):
        if not edges:
            return []
        # adjacency sets of the graph
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # remaining edges as a set for O(1) removal
        rem = set(edges)
        # degree within remaining subgraph
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1

        cliques = []
        # active vertices
        active = set(range(n))
        active = {v for v in range(n) if deg[v] > 0}

        while rem:
            if time.time() > deadline:
                break
            # Choose a seed vertex: high remaining degree to encourage large cliques
            # pick vertex with max deg among active
            seed_v = max(active, key=lambda x: deg[x]) if active else None
            if seed_v is None:
                # orphan edges: just take them as 2-cliques
                for (u, v) in list(rem):
                    cliques.append([u, v])
                rem.clear()
                break

            # Greedily grow a clique containing seed_v using Bron–Kerbosch-like
            # candidate ordering by common remaining neighbors.
            clique = [seed_v]
            candidates = set(adj[seed_v])
            # order candidates by how many shared neighbors they have with current clique
            def score(cand):
                # number of candidates that are adjacent to cand AND to all in clique
                s = 0
                for c2 in candidates:
                    if c2 == cand:
                        continue
                    if c2 in adj[cand] and all(c2 in adj[c] for c in clique):
                        s += 1
                return s
            # Build clique greedily
            while candidates:
                if time.time() > deadline:
                    break
                # pick best candidate
                best = max(candidates, key=score)
                # verify best is adjacent to all in clique
                if all(best in adj[c] for c in clique):
                    clique.append(best)
                    candidates.discard(best)
                else:
                    candidates.discard(best)
            # Ensure clique size >= 2; if only seed, add one neighbor
            if len(clique) < 2:
                # find any neighbor with a remaining edge
                for nb in adj[seed_v]:
                    if (min(seed_v, nb), max(seed_v, nb)) in rem:
                        clique.append(nb)
                        break
            if len(clique) < 2:
                # no valid edge from seed; remove seed from active
                active.discard(seed_v)
                continue

            # Remove all edges inside clique from rem, update deg
            removed = 0
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    e = (a, b) if a < b else (b, a)
                    if e in rem:
                        rem.discard(e)
                        deg[a] -= 1
                        deg[b] -= 1
                        removed += 1
            if removed == 0:
                # no edges removed; avoid infinite loop
                active.discard(seed_v)
                continue
            cliques.append(clique)
            # update active
            for v in clique:
                if deg[v] == 0:
                    active.discard(v)
            # prune active set occasionally
            if len(active) > 0:
                active = {v for v in active if deg[v] > 0}

        # Handle any leftover edges as 2-cliques (safety)
        for (u, v) in rem:
            cliques.append([u, v])
        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = instance(n, p, seed)
        cliques = partition(edges, n, _DEADLINE)
        results.append(cliques)
    return results
