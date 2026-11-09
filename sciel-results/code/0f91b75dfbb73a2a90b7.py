"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    from itertools import combinations

    t0 = time.time()
    DEADLINE = t0 + 110.0  # safety margin before 120s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Build adjacency sets for quick lookup
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)

        # For each independent-set vertex v (v >= k), compute its neighborhood in the clique part
        # and group by neighborhood
        from collections import defaultdict
        neigh = {}
        for v in range(k, n):
            ns = frozenset(u for u in range(k) if u in adj[v])
            neigh[v] = ns

        groups = defaultdict(list)
        for v in range(k, n):
            groups[neigh[v]].append(v)

        # Strategy:
        # 1. We need to cover all edges within the clique part (0..k-1) and all cross edges.
        # 2. For each group G with neighborhood S, we can form cliques that include vertices from G
        #    and subsets of S.
        # 3. Key insight: if we form a clique C = S ∪ G, this covers:
        #    - All edges within S (clique-clique edges)
        #    - All edges between S and G (cross edges)
        #    - No edges within G (they're independent)
        # 4. But we can't have the same clique-clique edge covered by multiple cliques.
        #
        # Better approach:
        # - For each group with non-empty neighborhood S, we need to cover all cross edges from S to G.
        # - We can use cliques of the form {u} ∪ G for each u in S. This covers all cross edges
        #   from u to all vertices in G, and no clique-clique edges.
        # - Then cover all clique-clique edges with one big clique of all k clique vertices.
        # - Total: sum over groups of |S| + 1
        #
        # Even better: if multiple groups have overlapping neighborhoods, we might merge them.
        # But for now, let's try a smarter grouping:
        # For each independent-set vertex v, its neighborhood is S_v.
        # If we have a set of independent vertices V' all with the same neighborhood S,
        # we can cover all cross edges from S to V' with |S| cliques: {u} ∪ V' for each u in S.
        #
        # Actually, let me think about this differently. The constraint is:
        # - Each edge must be in exactly one clique.
        # - Each clique must be a clique in the graph.
        #
        # For split graphs, the edges are:
        # 1. Edges within the clique part (all pairs of vertices in 0..k-1)
        # 2. Cross edges between the clique part and the independent set
        # 3. No edges within the independent set
        #
        # A clique in this graph can contain:
        # - Any subset of the clique part (since they're all connected)
        # - At most one independent-set vertex, OR multiple independent-set vertices
        #   that all have the same neighborhood containing all the clique vertices included.
        #
        # Wait, that's not right. If a clique contains independent-set vertices v1, v2,
        # then v1 and v2 must both be connected to all clique vertices in the clique.
        # So if the clique contains clique vertices S and independent vertices G,
        # then every v in G must have N(v) ⊇ S.
        #
        # So the valid cliques are: S ∪ G where S ⊆ [0,k) and G ⊆ {v >= k : S ⊆ N(v)}.
        #
        # To minimize the number of cliques, we want to make each clique as large as possible.
        #
        # Greedy approach:
        # 1. Start with all edges uncovered.
        # 2. Repeatedly find the largest valid clique that covers the most uncovered edges.
        # 3. Remove those edges and repeat.
        #
        # But this could be slow. Let's think about the structure more.
        #
        # Observation: The clique-clique edges (within 0..k-1) form a complete graph K_k.
        # We need to partition these edges into cliques. The minimum number of cliques to
        # partition K_k is... well, we can use one clique of all k vertices, which covers
        # all k(k-1)/2 edges. But then we can't use any other clique that contains two
        # clique vertices, because those edges would be covered twice.
        #
        # Wait, that's the issue. If we use the big clique [0..k-1], then no other clique
        # can contain two vertices from [0..k-1]. So any other clique can contain at most
        # one vertex from [0..k-1].
        #
        # So the strategy is:
        # 1. Use one clique: [0, 1, ..., k-1]. This covers all clique-clique edges.
        # 2. For each independent-set vertex v, we need to cover its cross edges.
        #    Since we can't use cliques with two clique vertices, each clique for cross
        #    edges can contain at most one clique vertex.
        # 3. For a group of independent vertices G with neighborhood S, we can cover all
        #    cross edges from u to G (for a fixed u in S) with one clique {u} ∪ G.
        #    So we need |S| such cliques for each group.
        #
        # But can we do better? What if we DON'T use the big clique?
        # Then we can use cliques with multiple clique vertices, as long as the independent
        # vertices in the clique are connected to all of them.
        #
        # Alternative strategy:
        # For each neighborhood S (a subset of [0,k)), let G_S be the set of independent
        # vertices with exactly neighborhood S.
        # We can form the clique S ∪ G_S. This covers:
        # - All edges within S
        # - All edges between S and G_S
        #
        # If we do this for each non-empty S, we cover:
        # - All cross edges (each cross edge (u,v) is covered by the clique for S = N(v))
        # - Some clique-clique edges: specifically, edges within each S
        #
        # The remaining clique-clique edges are those pairs (u,u') where u,u' < k
        # and there's no independent vertex v with {u,u'} ⊆ N(v).
        # For these, we need additional cliques.
        #
        # This might be better than the big clique approach when neighborhoods are
        # "large" and cover most pairs.
        #
        # Let's implement both and pick the better one.

        def strategy_big_clique(groups):
            """Use one big clique for clique-clique edges, then |S| cliques per group."""
            cliques = []
            cliques.append(list(range(k)))
            for S, G in groups.items():
                if not S:
                    continue
                for u in S:
                    cliques.append([u] + G)
            return cliques

        def strategy_neighborhood_cliques(groups, k):
            """Use S ∪ G for each non-empty neighborhood S, plus cliques for uncovered clique edges."""
            cliques = []
            covered_c_edges = set()
            for S, G in groups.items():
                if not S:
                    continue
                clique = list(S) + G
                cliques.append(clique)
                # Mark all edges within S as covered
                for (a, b) in combinations(S, 2):
                    covered_c_edges.add((min(a, b), max(a, b)))

            # Find uncovered clique-clique edges
            uncovered = []
            for u in range(k):
                for v in range(u + 1, k):
                    if (u, v) not in covered_c_edges:
                        uncovered.append((u, v))

            if not uncovered:
                return cliques

            # Need to cover remaining clique-clique edges with cliques of clique vertices only
            # Greedy: find largest clique that covers most uncovered edges
            # Since all clique vertices are mutually connected, any subset is a clique.
            # We need to partition the uncovered edges into cliques.
            # This is equivalent to: find a set of subsets of [0,k) such that every
            # uncovered edge is in exactly one subset.
            # Greedy: repeatedly pick the largest set of vertices that has many uncovered edges.
            # Actually, since any subset of clique vertices forms a clique, we just need to
            # partition the edge set. The minimum number of cliques to partition a graph is
            # the chromatic number of the complement... this is complex.
            # Simple approach: use one clique per uncovered edge (worst case).
            # Better: group uncovered edges.
            # Since the uncovered edges form a subgraph of K_k, we can try to find
            # a few large cliques that cover many edges.
            # Greedy: while uncovered edges remain, find the pair (u,v) with the most
            # common neighbors among uncovered edges, and form a clique.
            # Actually, let's just use the simple approach: one clique per edge if needed,
            # but try to merge.
            # For now, use a greedy set cover approach.
            remaining = set(uncovered)
            while remaining:
                # Find the vertex with the most uncovered edges
                best_vertex = None
                best_count = -1
                for (a, b) in remaining:
                    for v in (a, b):
                        count = sum(1 for (x, y) in remaining if x == v or y == v)
                        if count > best_count:
                            best_count = count
                            best_vertex = v
                if best_vertex is None:
                    break
                # Find all vertices connected to best_vertex via uncovered edges
                connected = set()
                for (a, b) in remaining:
                    if a == best_vertex:
                        connected.add(b)
                    elif b == best_vertex:
                        connected.add(a)
                # The clique is {best_vertex} ∪ connected
                clique = [best_vertex] + sorted(connected)
                cliques.append(clique)
                # Remove all edges within this clique
                clique_set = set(clique)
                new_remaining = set()
                for (a, b) in remaining:
                    if a in clique_set and b in clique_set:
                        continue  # covered
                    else:
                        new_remaining.add((a, b))
                remaining = new_remaining

            return cliques

        # Choose the better strategy
        if time.time() > DEADLINE:
            return strategy_big_clique(groups)

        cliques1 = strategy_big_clique(groups)
        cliques2 = strategy_neighborhood_cliques(groups, k)

        if len(cliques1) <= len(cliques2):
            return cliques1
        else:
            return cliques2

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            break
        cliques = partition_cliques(*spec)
        # Ensure every clique is a list of distinct Python ints, len >= 2
        cleaned = []
        for c in cliques:
            cl = [int(x) for x in c]
            if len(cl) >= 2 and len(set(cl)) == len(cl):
                cleaned.append(cl)
        result.append(cleaned)

    return result
# EVOLVE-BLOCK-END
