def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)

        # For each independent set vertex v, N(v) = neighbors in [0,k)
        neighbor_sets = {}
        for v in range(k, n):
            neighbors = [u for u in range(k) if (u, v) in edge_set]
            neighbor_sets[v] = set(neighbors)

        # We want to select a set S of independent-set vertices such that:
        # - The cliques {v} U N(v) for v in S are valid cliques (they are, by construction)
        # - No edge is covered by more than one clique
        #
        # Edges covered by clique {v} U N(v):
        #   - All cross edges (u, v) for u in N(v)
        #   - All internal edges (u1, u2) for u1, u2 in N(v)
        #
        # Constraint: For any two selected vertices v, w:
        #   - They must not share a cross edge: impossible since cross edges are (u,v) and (u,w) with different second components
        #   - They must not share an internal edge: i.e., the set of internal edges in N(v) and N(w) must be disjoint
        #     This means: for any internal edge (u1, u2) in the clique graph, it cannot be that both u1, u2 are in N(v) AND both u1, u2 are in N(w)
        #     Equivalently: N(v) and N(w) cannot both contain an edge of the clique graph
        #     More precisely: the set of edges of the clique graph induced by N(v) and the set induced by N(w) must be disjoint
        #
        # Also, if an internal edge (u1, u2) is covered by clique for v, it cannot be covered by any other clique.
        # Cross edges are unique to each v, so no conflict there.
        #
        # So the constraint is: for selected vertices v1, v2, ..., vm, the edge sets E(N(vi)) are pairwise disjoint.
        #
        # Additionally, we need to cover ALL edges. The remaining edges (not covered by selected cliques) will be 2-cliques.
        # To minimize total cliques, we want to maximize the number of edges covered by the selected large cliques,
        # while respecting the disjointness constraint.
        #
        # Greedy approach:
        # 1. For each v in independent set, compute the set of internal edges covered by {v} U N(v).
        # 2. Sort vertices by the number of edges they cover (cross + internal) in descending order.
        # 3. Greedily select vertices if their internal edge set is disjoint from all previously selected.
        # 4. The selected vertices give large cliques. All remaining edges become 2-cliques.

        # Compute for each v the set of internal edges covered
        internal_edge_sets = {}
        for v in range(k, n):
            S = neighbor_sets[v]
            if len(S) < 2:
                internal_edge_sets[v] = set()
            else:
                S_list = sorted(S)
                edges_v = set()
                for i in range(len(S_list)):
                    for j in range(i + 1, len(S_list)):
                        u1, u2 = S_list[i], S_list[j]
                        if (u1, u2) in edge_set:
                            edges_v.add((u1, u2))
                internal_edge_sets[v] = edges_v

        # Total edges covered by clique for v = |N(v)| cross edges + |internal_edge_sets[v]| internal edges
        def total_covered(v):
            return len(neighbor_sets[v]) + len(internal_edge_sets[v])

        # Sort independent set vertices by total covered edges, descending
        indep_vertices = list(range(k, n))
        indep_vertices.sort(key=lambda v: total_covered(v), reverse=True)

        selected = []
        used_internal_edges = set()

        for v in indep_vertices:
            ie = internal_edge_sets[v]
            # Check if any internal edge in ie is already used
            if ie & used_internal_edges:
                continue
            # Select v
            selected.append(v)
            used_internal_edges |= ie

        # Build the clique list
        cliques = []

        # Add large cliques for selected vertices
        for v in selected:
            S = neighbor_sets[v]
            if len(S) >= 1:
                clique = sorted(list(S) + [v])
                cliques.append(clique)

        # Determine which edges are covered by the large cliques
        covered_edges = set()
        for v in selected:
            S = neighbor_sets[v]
            # Cross edges
            for u in S:
                covered_edges.add((u, v))
            # Internal edges
            covered_edges |= internal_edge_sets[v]

        # Remaining edges become 2-cliques
        for e in sorted(edge_set):
            if e not in covered_edges:
                cliques.append([e[0], e[1]])

        return [list(c) for c in cliques]

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    return [clique_partition(*spec) for spec in specs]
