import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def popcount(x):
        return bin(x).count('1')

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Build adjacency as bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # Track remaining edges as a set of tuples
        remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)

        cliques = []

        def build_active_adj(remaining_edges, n):
            """Build active adjacency bitmasks from remaining edges."""
            active_adj = [0] * n
            for u, v in remaining_edges:
                active_adj[u] |= (1 << v)
                active_adj[v] |= (1 << u)
            return active_adj

        def greedy_clique_from(active_adj, start, n):
            """Greedy clique construction starting from a given vertex."""
            clique = [start]
            P = active_adj[start] & ~(1 << start)
            while P:
                # Pick vertex in P with max degree in P
                best_v = -1
                best_d = -1
                tmp = P
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    d = popcount(active_adj[v] & P)
                    if d > best_d:
                        best_d = d
                        best_v = v
                    tmp ^= lsb
                if best_v < 0:
                    break
                clique.append(best_v)
                P = P & active_adj[best_v]
                P &= ~(1 << best_v)
            return clique if len(clique) >= 2 else None

        def find_best_clique_by_edge_coverage(active_adj, n, remaining_edges):
            """
            Evaluate cliques rooted at every active vertex, pick the one that covers
            the most currently uncovered edges. This is a best-first strategy.
            """
            active_vertices = [v for v in range(n) if active_adj[v] != 0]
            if not active_vertices:
                return None

            # Sort by degree descending
            active_vertices.sort(key=lambda v: popcount(active_adj[v]), reverse=True)

            # Limit the number of candidates to evaluate
            limit = min(len(active_vertices), 80)
            candidates = active_vertices[:limit]

            best_clique = None
            best_coverage = 0

            for start in candidates:
                if time.time() > deadline - 25:
                    break
                clique = greedy_clique_from(active_adj, start, n)
                if clique is None:
                    continue

                # Count how many remaining edges are covered by this clique
                clique_set = set(clique)
                coverage = 0
                for u, v in remaining_edges:
                    if u in clique_set and v in clique_set:
                        coverage += 1

                if coverage > best_coverage:
                    best_coverage = coverage
                    best_clique = clique

            return best_clique

        while remaining_edges:
            if time.time() > deadline - 20:
                # Emergency: pack remaining edges as pairs
                rem_list = list(remaining_edges)
                for e in rem_list:
                    cliques.append(list(e))
                break

            # Build active adjacency from remaining edges
            active_adj = build_active_adj(remaining_edges, n)

            # Use best-first selection: evaluate cliques from multiple starts
            clique = find_best_clique_by_edge_coverage(active_adj, n, remaining_edges)

            if clique is None:
                # Fallback: take one edge
                e = next(iter(remaining_edges))
                cliques.append(list(e))
                remaining_edges.discard(e)
                continue

            # Remove all edges within the clique from remaining_edges
            clique_set = set(clique)
            to_remove = set()
            for u, v in remaining_edges:
                if u in clique_set and v in clique_set:
                    to_remove.add((u, v))

            if to_remove:
                cliques.append(clique)
                remaining_edges.difference_update(to_remove)
            else:
                # Shouldn't happen if clique is valid, but safety fallback
                e = next(iter(remaining_edges))
                cliques.append(list(e))
                remaining_edges.discard(e)

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
