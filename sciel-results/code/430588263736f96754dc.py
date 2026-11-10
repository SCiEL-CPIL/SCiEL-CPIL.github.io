"""Rewritten program for clique partition of split graphs."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # C vertices: 0..k-1, I vertices: k..n-1
        C = list(range(k))
        I = list(range(k, n))

        cliques = []

        # For each independent vertex v, find its neighbors in C
        # Group C-I edges into cliques: {v, u} or {v, u1, u2} where u1,u2 are both neighbors of v
        # Strategy: for each v in I, pair up its C-neighbors into pairs, each pair + v forms a triangle (clique of size 3)
        # If odd number of neighbors, one forms a clique of size 2 with v

        for v in I:
            neighbors_in_C = [u for u in C if u in adj[v]]
            # Pair up neighbors
            i = 0
            while i < len(neighbors_in_C) - 1:
                u1 = neighbors_in_C[i]
                u2 = neighbors_in_C[i + 1]
                # {v, u1, u2} is a clique since u1-u2 edge exists (both in C) and both connect to v
                cliques.append([u1, u2, v])
                i += 2
            if i < len(neighbors_in_C):
                u1 = neighbors_in_C[i]
                cliques.append([u1, v])

        # Now cover C-C edges
        # Remaining C-C edges not yet covered by the triangles above
        # Each triangle {u1, u2, v} covers edge (u1, u2)
        covered_cc = set()
        for c in cliques:
            if len(c) == 3:
                u1, u2, v = c
                if u1 < k and u2 < k:
                    covered_cc.add((min(u1, u2), max(u1, u2)))
            # size-2 cliques are C-I, don't cover C-C edges

        # Find uncovered C-C edges
        all_cc_edges = set()
        for u in C:
            for v in C:
                if u < v and v in adj[u]:
                    all_cc_edges.add((u, v))
        uncovered_cc = all_cc_edges - covered_cc

        # Cover remaining C-C edges: pair them up into cliques of size 2
        # But we need cliques of size >= 2, so each uncovered edge becomes its own clique
        # Better: try to group uncovered C-C edges into larger cliques
        # Since C is a clique, any subset of C forms a clique
        # We want to partition uncovered_cc edges into minimum number of cliques
        # This is edge coloring of a graph, but simpler: since all vertices in C form a clique,
        # we can cover edges greedily by finding maximal cliques in the remaining edge set

        # Build remaining edge set on C vertices
        rem_edges = list(uncovered_cc)
        if not rem_edges:
            return cliques

        # Greedy: repeatedly find a max clique in the remaining edge subgraph
        rem_adj = defaultdict(set)
        for u, v in rem_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        rem_vertices = set(rem_adj.keys())

        while rem_vertices:
            # Find a greedy max clique in the remaining graph
            # Start with a vertex of max degree, expand
            if not rem_vertices:
                break
            # Pick vertex with max degree in remaining
            start = max(rem_vertices, key=lambda x: len(rem_adj.get(x, set())))
            clique = {start}
            candidates = set(rem_adj.get(start, set())) & rem_vertices
            # Greedily add vertices
            while True:
                best = None
                best_score = -1
                for cand in candidates:
                    # Score: how many edges it adds
                    score = len(candidates & rem_adj.get(cand, set()))
                    if score > best_score:
                        best_score = score
                        best = cand
                if best is None or best_score < 0:
                    break
                clique.add(best)
                candidates = candidates & rem_adj.get(best, set())

            if len(clique) >= 2:
                cliques.append(sorted(clique))
                # Remove all edges within this clique from remaining
                clique_list = sorted(clique)
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        u, v = clique_list[i], clique_list[j]
                        if (min(u, v), max(u, v)) in rem_adj:
                            rem_adj[u].discard(v)
                            rem_adj[v].discard(u)
                # Remove vertices with no remaining edges
                for x in list(rem_vertices):
                    if not rem_adj.get(x, set()):
                        rem_vertices.discard(x)
            else:
                # Should not happen since we have edges
                break

        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        cliques = partition_split_graph(n, k, p, seed)
        # Validate: ensure all vertices in cliques are Python int
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(v) for v in c])
        results.append(clean_cliques)

    return results
# EVOLVE-BLOCK-END
