"""Improved program for clique partitioning of dense random graphs."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, edges, remaining_time):
        # Build adjacency sets for fast lookup
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))

        # We need to partition all edges into cliques (each clique >= 2 vertices)
        # Strategy: greedily find large cliques that cover many uncovered edges,
        # then handle remaining edges as individual pairs.

        # Build a set of uncovered edges
        uncovered = set(edge_set)
        cliques = []

        # Precompute: for each pair of vertices, check if they form a triangle with a third
        # We'll use a greedy approach: repeatedly find the largest clique among uncovered edges

        # For efficiency, let's try to find cliques by looking at vertices with high degree in uncovered subgraph
        # A clique of size k covers C(k,2) edges.

        # Greedy: while there are uncovered edges, find the best clique to add
        # "Best" = maximizes (number of uncovered edges covered) / (clique size - 1)
        # Actually, we want to minimize total number of cliques, so we want large cliques.

        # Let's try a different approach: 
        # 1. Find all triangles in the graph
        # 2. Use a greedy set cover / matching approach to combine edges into cliques

        # Simpler greedy: 
        # While uncovered edges remain:
        #   Find a vertex v with maximum uncovered degree
        #   Among its uncovered neighbors, find the largest clique
        #   Add that clique (including v) to the partition
        #   Remove all edges in that clique from uncovered
        #   If no clique of size >= 2 found (shouldn't happen since v has uncovered neighbors), 
        #     just take one edge

        # Even simpler and effective for dense graphs:
        # Try to find large cliques using a greedy expansion

        def find_clique_from_start(start_vertex, candidate_set, max_size=10):
            """Find a clique containing start_vertex using greedy expansion."""
            clique = [start_vertex]
            candidates = set(candidate_set)
            candidates.discard(start_vertex)
            
            # Greedily add vertices that are connected to all current clique members
            while candidates:
                # Find the candidate with highest degree within candidates that connects to all clique members
                best = None
                best_score = -1
                for c in candidates:
                    if all(c in adj[v] for v in clique):
                        # Score by how many other candidates it connects to
                        score = sum(1 for d in candidates if d != c and d in adj[c])
                        if score > best_score:
                            best_score = score
                            best = c
                if best is None:
                    break
                clique.append(best)
                candidates.discard(best)
                if len(clique) >= max_size:
                    break
            
            return clique

        # Greedy clique extraction
        max_clique_size = 12
        iteration = 0
        while uncovered:
            if time.time() > remaining_time:
                break
            iteration += 1

            # Find vertex with max uncovered degree
            uncovered_deg = {}
            for (u, v) in uncovered:
                uncovered_deg[u] = uncovered_deg.get(u, 0) + 1
                uncovered_deg[v] = uncovered_deg.get(v, 0) + 1

            if not uncovered_deg:
                break

            # Pick vertex with highest uncovered degree
            start_v = max(uncovered_deg, key=lambda x: uncovered_deg[x])

            # Get neighbors of start_v in uncovered subgraph
            neighbors = set()
            for (u, v) in uncovered:
                if u == start_v:
                    neighbors.add(v)
                elif v == start_v:
                    neighbors.add(u)

            if not neighbors:
                # Shouldn't happen, but safety
                break

            # Find a clique containing start_v using greedy expansion
            # Candidates are neighbors that are connected to start_v (all are, by definition)
            # and we need them to be mutually connected
            clique = find_clique_from_start(start_v, neighbors, max_clique_size)

            # Verify it's actually a clique and all edges are uncovered
            valid = len(clique) >= 2
            if valid:
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (a, b) if a < b else (b, a)
                        if e not in uncovered:
                            valid = False
                            break
                    if not valid:
                        break

            if not valid or len(clique) < 2:
                # Just take one edge
                e = next(iter(uncovered))
                cliques.append(list(e))
                uncovered.discard(e)
            else:
                cliques.append(clique)
                # Remove all edges in this clique from uncovered
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (a, b) if a < b else (b, a)
                        uncovered.discard(e)

        # Any remaining uncovered edges become individual 2-cliques
        for e in uncovered:
            if time.time() > remaining_time:
                break
            cliques.append(list(e))

        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Fallback: just return individual edges
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue

        edges = instance(n, p, seed)
        remaining = deadline - time.time()
        if remaining < 5:
            results.append([list(e) for e in edges])
            continue

        cliques = solve_instance(n, edges, remaining)
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
