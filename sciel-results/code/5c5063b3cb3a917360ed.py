"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        rng = Random(seed)
        # Build adjacency
        adj = [set() for _ in range(n)]
        # C is 0..k-1, I is k..n-1
        for u in range(k):
            for v in range(u+1, k):
                adj[u].add(v)
                adj[v].add(u)
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)

        # For each I-vertex, get its neighbors in C
        i_neighbors = {}
        for v in range(k, n):
            nbrs = [u for u in range(k) if u in adj[v]]
            if nbrs:
                i_neighbors[v] = nbrs

        cliques = []
        covered_edges = set()

        # Step 1: For each I-vertex with edges, create clique {v} + N_C(v)
        for v in sorted(i_neighbors.keys()):
            clique = [v] + i_neighbors[v]
            cliques.append(clique)
            # Mark all edges in this clique as covered
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    a, b = clique[i], clique[j]
                    if a < b:
                        covered_edges.add((a, b))
                    else:
                        covered_edges.add((b, a))

        # Step 2: Find remaining C-internal edges not covered
        remaining = []
        for u in range(k):
            for v in range(u+1, k):
                if (u, v) not in covered_edges:
                    remaining.append((u, v))

        # Cover remaining C-internal edges with cliques from C
        # Since C is a clique, any subset is a clique. We want to cover remaining edges with min cliques.
        # This is equivalent to: we have a graph on C-vertices with edges = remaining, cover edges with cliques.
        # Since the underlying graph is complete, any subset of vertices forms a clique.
        # So we need to partition the edge set into cliques = cover edges with complete subgraphs.
        # Greedy: repeatedly pick a maximum clique in the remaining-edge graph.

        # Build remaining edge set for C
        rem_adj = [set() for _ in range(k)]
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)

        # Tomita's maximum clique on the remaining-edge graph (which is a subgraph of complete graph on k vertices)
        def max_clique(vertices, adj_sets):
            """Find maximum clique in graph defined by adj_sets on given vertices."""
            if not vertices:
                return []
            # Greedy + branch and bound
            best = []
            candidates = list(vertices)

            def expand(clique, cand):
                nonlocal best
                if not cand:
                    if len(clique) > len(best):
                        best = list(clique)
                    return
                if len(clique) + len(cand) <= len(best):
                    return
                # Pivot: pick vertex with max degree in cand
                pivot = max(cand, key=lambda v: len(adj_sets[v] & set(cand)))
                # Branch on candidates not adjacent to pivot
                for v in list(cand):
                    if v not in adj_sets[pivot]:
                        new_clique = clique + [v]
                        new_cand = [u for u in cand if u in adj_sets[v]]
                        expand(new_clique, new_cand)
                        cand.remove(v)
                        if len(clique) + len(cand) <= len(best):
                            return

            expand([], candidates)
            return best

        # Iteratively extract maximum cliques from remaining edges
        while remaining:
            # Build vertex set of remaining
            verts = set()
            for u, v in remaining:
                verts.add(u)
                verts.add(v)
            mc = max_clique(verts, rem_adj)
            if len(mc) < 2:
                # Shouldn't happen if remaining has edges
                break
            cliques.append(mc)
            # Remove edges in mc from remaining
            mc_set = set(mc)
            new_remaining = []
            for u, v in remaining:
                if u not in mc_set or v not in mc_set:
                    new_remaining.append((u, v))
            remaining = new_remaining
            # Update rem_adj
            for u in range(k):
                rem_adj[u] = rem_adj[u] - mc_set

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))

    # Convert to ensure types
    final = []
    for graph_cliques in results:
        g = []
        for c in graph_cliques:
            g.append([int(v) for v in c])
        final.append(g)
    return final
# EVOLVE-BLOCK-END

