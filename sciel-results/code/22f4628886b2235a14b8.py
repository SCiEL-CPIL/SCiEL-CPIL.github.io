import time
import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    results = []
    start_time = time.time()

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        # Reconstruct edges
        rng = random.Random(seed)
        neighbors = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                neighbors[u].add(v)
                neighbors[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[u].add(v)
                    neighbors[v].add(u)

        cliques = []
        covered_edges_in_C = set()

        # Get I vertices with their C-neighborhoods
        I_vertices = []
        for v in range(k, n):
            N_v = neighbors[v] & set(range(k))
            if N_v:
                I_vertices.append((v, N_v))

        # Sort by a heuristic: prefer vertices whose N_v has fewer internal conflicts
        # with already-covered edges. Initially none covered, so sort by |N_v| descending
        # to pack large cliques first.
        I_vertices.sort(key=lambda x: len(x[1]), reverse=True)

        for v, N_v in I_vertices:
            N_v_sorted = sorted(N_v)
            # Find the largest subset of N_v that has no pair in covered_edges_in_C
            # Greedy: try to include all, remove conflicting vertices
            # Build a conflict graph on N_v where edge (a,b) exists if (a,b) in covered_edges_in_C
            # We want a maximum independent set in this conflict graph.
            # For small N_v, we can do this exactly or with a good heuristic.

            # Simple approach: greedily build the subset
            # Start with all vertices, remove the one involved in most conflicts
            subset = list(N_v_sorted)
            changed = True
            while changed:
                changed = False
                # Find vertex with most conflicts within current subset
                max_conflicts = -1
                max_vertex = None
                for i, a in enumerate(subset):
                    conflicts = 0
                    for j, b in enumerate(subset):
                        if i != j:
                            if (min(a, b), max(a, b)) in covered_edges_in_C:
                                conflicts += 1
                    if conflicts > max_conflicts:
                        max_conflicts = conflicts
                        max_vertex = a
                if max_conflicts > 0 and max_vertex is not None:
                    subset.remove(max_vertex)
                    changed = True

            if subset:
                subset.sort()
                cliques.append([v] + subset)
                for u1, u2 in combinations(subset, 2):
                    covered_edges_in_C.add((u1, u2))
            else:
                # No valid subset found (shouldn't happen since single vertices are always valid)
                # Fall back to singletons
                for u in N_v_sorted:
                    cliques.append([v, u])

        # Handle remaining C-edges
        remaining_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_edges_in_C:
                    remaining_edges.add((u, w))

        # Greedy maximal clique extraction for remaining C-edges
        rem_set = set(remaining_edges)
        while rem_set:
            # Pick an edge with minimum degree in the remaining edge graph
            # to encourage larger cliques
            min_degree = float('inf')
            chosen_edge = None
            for edge in rem_set:
                u, w = edge
                deg = 0
                for x in range(k):
                    if x != u and x != w:
                        if (min(u, x), max(u, x)) in rem_set and (min(w, x), max(w, x)) in rem_set:
                            deg += 1
                if deg < min_degree:
                    min_degree = deg
                    chosen_edge = edge

            if chosen_edge is None:
                break

            u, w = chosen_edge
            clique_vertices = {u, w}
            # Try to add other vertices that connect to all in clique_vertices
            candidates = []
            for x in range(k):
                if x not in clique_vertices:
                    is_connected = True
                    for y in clique_vertices:
                        if (min(x, y), max(x, y)) not in rem_set:
                            is_connected = False
                            break
                    if is_connected:
                        candidates.append(x)

            clique_vertices.update(candidates)

            # Remove all edges within clique_vertices from rem_set
            for u1, u2 in combinations(sorted(clique_vertices), 2):
                rem_set.discard((u1, u2))

            if len(clique_vertices) >= 2:
                cliques.append(list(clique_vertices))

        # Convert to list of lists of ints
        results.append(cliques)

        if time.time() - start_time > 100:
            break

    return results

