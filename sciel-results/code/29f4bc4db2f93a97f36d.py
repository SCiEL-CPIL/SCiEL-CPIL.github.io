"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import defaultdict

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        # Separate vertices
        K = list(range(k))  # clique part
        I = list(range(k, n))  # independent set part
        
        # Build adjacency for independent vertices to clique vertices
        # neigh[v] = bitmask of neighbors in K
        # Map K vertex to bit position
        k_to_bit = {u: i for i, u in enumerate(K)}
        
        neigh_mask = {}
        for v in I:
            mask = 0
            for u in K:
                if (min(u, v), max(u, v)) in edge_set:
                    mask |= (1 << k_to_bit[u])
            neigh_mask[v] = mask
        
        # Group independent vertices by their exact neighborhood mask
        groups = defaultdict(list)
        for v in I:
            groups[neigh_mask[v]].append(v)
        
        # We need to partition all edges into cliques.
        # Edges in K (clique part): all pairs (u, v) with u, v in K.
        # Cross edges: (u, v) with u in K, v in I.
        
        # Strategy:
        # 1. The clique K can be covered by one clique K itself.
        # 2. For cross edges, we group I by neighborhood. 
        #    If we use a clique C = S ∪ T where S ⊆ K and T ⊆ I, 
        #    then for C to be a clique, every v in T must be adjacent to every u in S.
        #    Also, no two vertices in T can be adjacent (they aren't, since I is independent).
        #    And all pairs in S must be adjacent (they are, since K is a clique).
        #    
        #    The constraint "every edge in exactly one clique" means:
        #    - K-K edges: covered by the K clique (if we use it) or distributed.
        #    - Cross edges: (u, v) covered by the clique that contains both u and v.
        #    
        #    If we use the full K as a clique, then no other clique can contain two vertices from K.
        #    So cross-edge cliques can contain at most one vertex from K? 
        #    No, they can contain multiple vertices from K, but then those K-K edges would be in that clique, 
        #    not in the K clique. This violates the partition if the K clique also covers them.
        #    
        #    So, if we use the full K as a clique, then every other clique can contain at most one vertex from K.
        #    This means each cross-edge clique is of the form {u} ∪ T where u ∈ K and T ⊆ I, 
        #    and every v in T is adjacent to u.
        #    
        #    But this is inefficient. We can do better by not using the full K as a clique, 
        #    but instead distributing K-K edges across the cross-edge cliques.
        
        # Let's try a different approach:
        # For each independent vertex v, it has a neighborhood N(v) ⊆ K.
        # All cross edges incident to v are (u, v) for u in N(v).
        # We can group independent vertices with the same neighborhood.
        # For a group G with neighborhood S, the set S ∪ G is a clique.
        # This clique covers:
        #   - All cross edges between S and G.
        #   - All K-K edges within S.
        
        # If we use one clique per distinct non-empty neighborhood S, 
        # then all cross edges are covered.
        # The K-K edges covered are those within the union of all S's.
        # But K-K edges not in any S (i.e., pairs of vertices in K that are not in any neighborhood) 
        # are not covered.
        
        # Also, if a K-K edge is in multiple S's, it's covered multiple times, which is not allowed.
        
        # So we need to partition the K-K edges as well.
        
        # Let's use a greedy approach:
        # 1. Start with all edges uncovered.
        # 2. Repeatedly find a maximal clique in the remaining graph, add it to the partition, 
        #    and remove its edges.
        
        # For efficiency, we can use the structure:
        # - The graph is a split graph.
        # - Cliques are either subsets of K, or S ∪ T where S ⊆ K, T ⊆ I, and every v in T is adjacent to every u in S.
        
        # Let's try:
        # 1. For each independent vertex v, consider the clique {v} ∪ N(v). 
        #    But this might be too small.
        # 2. Group by neighborhood: for each distinct S, use S ∪ G_S where G_S is the group with neighborhood S.
        #    This covers all cross edges for that group.
        #    The K-K edges within S are also covered.
        # 3. After covering all cross edges, some K-K edges might remain uncovered.
        #    Cover them with additional cliques (subsets of K).
        
        # But we must ensure no K-K edge is covered twice.
        
        # Let's track which K-K edges are covered by the group cliques.
        # For each group with neighborhood S, the K-K edges within S are covered.
        # If two groups have neighborhoods S1 and S2 with S1 ∩ S2 non-empty, 
        # then K-K edges in S1 ∩ S2 are covered twice.
        
        # To avoid this, we need to ensure that the neighborhoods S are disjoint? 
        # But they are not necessarily disjoint.
        
        # Alternative: 
        # For each independent vertex v, assign it to exactly one clique.
        # The clique containing v must include v and some subset of N(v).
        # To maximize the size of the clique, we can include all of N(v).
        # But if multiple v's are in the same clique, they must share the same neighborhood.
        
        # So, group by exact neighborhood. For each group, use one clique S ∪ G.
        # This is valid if no two groups have overlapping neighborhoods in a way that causes double coverage of K-K edges.
        
        # Actually, the K-K edges within S are covered by the clique S ∪ G.
        # If another group has neighborhood S' that overlaps with S, then K-K edges in S ∩ S' are covered twice.
        
        # So this approach only works if the neighborhoods are disjoint, which is not generally true.
        
        # Let's use a different strategy:
        # 1. For each independent vertex v, create a "star" clique: {v} ∪ N(v).
        #    This covers all cross edges incident to v and all K-K edges within N(v).
        #    But this uses one clique per independent vertex, which is too many.
        
        # 2. Merge stars: if two independent vertices v1 and v2 have N(v1) ⊆ N(v2), 
        #    we can put them in the same clique? No, because then the clique would need to contain 
        #    N(v1) ∪ N(v2) = N(v2), and v1 must be adjacent to all in N(v2), which is not guaranteed.
        
        # 3. Use the exact neighborhood grouping, but handle K-K edge conflicts by adjusting.
        
        # Given the complexity, let's use a general greedy clique partition:
        # 1. Start with all edges uncovered.
        # 2. While there are uncovered edges:
        #    a. Pick an uncovered edge (u, v).
        #    b. Find the maximal clique containing (u, v) in the remaining graph.
        #    c. Add this clique to the partition and remove its edges.
        
        # For efficiency, we can precompute candidate cliques.
        
        # Let's try a simpler heuristic:
        # 1. Group independent vertices by exact neighborhood.
        # 2. For each group with non-empty neighborhood S, create a clique S ∪ G.
        # 3. Track which K-K edges are covered by these cliques.
        # 4. For any remaining K-K edges, cover them with additional cliques.
        
        # To handle the double coverage issue, we can be more careful:
        # When we create a clique S ∪ G, we mark all K-K edges within S as covered.
        # If a K-K edge is already covered, we don't include it in the new clique? 
        # But the clique is defined by its vertices, not edges. 
        # The constraint is that every edge is in exactly one clique.
        
        # So if we have two cliques C1 and C2 that both contain vertices a, b ∈ K, 
        # then the edge (a, b) is in both C1 and C2, which is not allowed.
        
        # Therefore, no two cliques can share two vertices from K.
        
        # This means that if we use cliques S ∪ G, the sets S must be pairwise disjoint (in terms of K vertices).
        
        # This is a strong constraint. It means we can't use this approach directly.
        
        # Let's try a different approach:
        # For each independent vertex v, we must cover its cross edges.
        # Each cross edge (u, v) must be in a clique that contains u and v.
        # Such a clique can contain other independent vertices w that are adjacent to u.
        # And it can contain other clique vertices that are adjacent to v.
        
        # Let's use a greedy edge-based approach, but optimized for split graphs.
        
        uncovered = set(edge_set)
        cliques = []
        
        # Precompute adjacency for quick lookup
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Sort edges for determinism
        sorted_edges = sorted(uncovered)
        
        while uncovered:
            # Pick the first uncovered edge
            u, v = sorted_edges[0]
            # Remove it from sorted_edges? No, we'll check membership.
            
            # Find the maximal clique containing u and v in the remaining graph
            # Start with {u, v}
            clique_vertices = {u, v}
            
            # Try to add more vertices
            changed = True
            while changed:
                changed = False
                # Try to add each vertex w not in clique_vertices
                for w in range(n):
                    if w in clique_vertices:
                        continue
                    # Check if w is adjacent to all vertices in clique_vertices
                    if all(w in adj[x] for x in clique_vertices):
                        # Check if all edges between w and clique_vertices are uncovered
                        # Actually, we need to ensure that adding w doesn't create a conflict.
                        # Since we're building a clique, all edges between w and clique_vertices must be in the graph.
                        # And they must be uncovered.
                        all_edges_uncovered = True
                        for x in clique_vertices:
                            e = (min(w, x), max(w, x))
                            if e not in uncovered:
                                all_edges_uncovered = False
                                break
                        if all_edges_uncovered:
                            clique_vertices.add(w)
                            changed = True
            
            # Add this clique
            clique_list = list(clique_vertices)
            if len(clique_list) >= 2:
                cliques.append(clique_list)
                # Remove all edges within this clique from uncovered
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        e = (min(clique_list[i], clique_list[j]), max(clique_list[i], clique_list[j]))
                        uncovered.discard(e)
            
            # Update sorted_edges (remove covered edges)
            sorted_edges = [e for e in sorted_edges if e in uncovered]
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve_instance(*spec))
    
    return results
# EVOLVE-BLOCK-END
