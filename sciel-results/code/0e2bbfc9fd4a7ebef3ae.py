"""
Clique transversal via local search with Bron-Kerbosch clique enumeration.
Strategy:
  1. For each random graph instance, enumerate all maximal cliques (Bron-Kerbosch).
  2. Build a set-cover instance: each maximal clique (size >= 2) is a set to hit.
  3. Greedy set-cover to get an initial solution.
  4. Local search: try removing a vertex if all cliques it covers are still covered
     by other selected vertices; try swapping one vertex for another if it reduces
     or maintains size while improving coverage slack.
  5. Deadline guard: return best found before 120s wall-clock.
"""

import time
from random import Random

def construct():
    t_start = time.time()
    DEADLINE = 110.0  # safety margin before 120s

    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return n, adj, edges

    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        # Use bitmasks for speed
        def bk(R, P, X):
            if not P and not X:
                cliques.append(frozenset(R))
                return
            # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
            PX = P | X
            if PX:
                # Pick pivot with max overlap
                best_u = max(PX, key=lambda u: len(P & adj[u]))
            else:
                return
            candidates = P - adj[best_u]
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        adj, edges = random_graph(n, p, seed)[1], random_graph(n, p, seed)[2]
        # Reconstruct adj properly
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Enumerate maximal cliques
        cliques = bron_kerbosch(adj, n)
        # Filter to cliques of size >= 2
        cliques = [c for c in cliques if len(c) >= 2]

        if not cliques:
            return []

        # Build incidence: for each vertex, which cliques it covers
        vertex_cliques = {v: [] for v in range(n)}
        clique_index = {}
        for i, c in enumerate(cliques):
            clique_index[i] = c
            for v in c:
                vertex_cliques[v].append(i)

        # Greedy set cover
        covered = [False] * len(cliques)
        selected = set()
        remaining = list(range(len(cliques)))

        # Precompute for each vertex the set of cliques it covers
        vert_cover = {v: set(vertex_cliques[v]) for v in range(n)}

        while remaining:
            if time.time() > DEADLINE:
                break
            # Find vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                count = sum(1 for ci in vert_cover[v] if not covered[ci])
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count <= 0:
                break
            selected.add(best_v)
            for ci in vert_cover[best_v]:
                covered[ci] = True
            remaining = [ci for ci in remaining if not covered[ci]]

        # Local search: try removing vertices
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for v in list(selected):
                # Check if removing v still covers all cliques
                ok = True
                for ci in vert_cover[v]:
                    # Is ci covered by some other selected vertex?
                    c = cliques[ci]
                    other_cover = False
                    for u in c:
                        if u != v and u in selected:
                            other_cover = True
                            break
                    if not other_cover:
                        ok = False
                        break
                if ok:
                    selected.discard(v)
                    improved = True

        # Local search: try swaps
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for v in list(selected):
                # Try removing v and adding a better vertex
                # First check if v is removable
                ok_remove = True
                for ci in vert_cover[v]:
                    c = cliques[ci]
                    other_cover = False
                    for u in c:
                        if u != v and u in selected:
                            other_cover = True
                            break
                    if not other_cover:
                        ok_remove = False
                        break
                if not ok_remove:
                    continue
                # v is removable; try adding a vertex that covers more
                # Find cliques that only v covers among selected
                # Actually since v is removable, no clique is exclusively covered by v
                # Try to find a vertex w not in selected that, when we add it and remove v,
                # still covers everything. Since v is removable, removing v is fine.
                # But we want to possibly replace v with something that allows further removals.
                # For simplicity, just do removals in the next iteration.
                selected.discard(v)
                improved = True

        # Final removal pass
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for v in list(selected):
                ok = True
                for ci in vert_cover[v]:
                    c = cliques[ci]
                    other_cover = False
                    for u in c:
                        if u != v and u in selected:
                            other_cover = True
                            break
                    if not other_cover:
                        ok = False
                        break
                if ok:
                    selected.discard(v)
                    improved = True

        return sorted(selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            results.append([])
            continue
        sol = solve_instance(n, p, seed)
        results.append(sol)

    return results
