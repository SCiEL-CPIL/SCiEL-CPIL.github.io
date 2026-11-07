"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from itertools import combinations

    def instance(n, k, p, seed):
        # Generate the split graph
        # First k vertices form a clique (K)
        # Remaining n-k vertices form an independent set (I)
        # Cross edges exist with probability p
        
        # For deterministic behavior, we need to generate edges in a specific order
        # The original uses: for u in range(k) for v in range(u+1, k) -> these are always present
        # Then: for v in range(k, n) for u in range(k) if rng.random() < p
        
        # Let's use a simple approach: generate all possible edges and filter
        # But we need to match the original generator's behavior
        
        import random
        rng = random.Random(seed)
        
        edges = set()
        
        # Clique edges (always present)
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        
        # Cross edges (probabilistic)
        # Order matters for reproducibility with the seed
        # Original: for v in range(k, n) for u in range(k)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((min(u, v), max(u, v)))
        
        # Independent set edges (never present)
        # No edges among vertices k..n-1
        
        return edges

    def solve_split_graph(n, k, p, seed):
        """Find minimum clique cover for a split graph."""
        edges = instance(n, k, p, seed)
        
        # K = {0, ..., k-1}, I = {k, ..., n-1}
        K = list(range(k))
        I = list(range(k, n))
        
        # Build adjacency for K vertices (within K, all connected)
        # and for cross edges
        
        # For each I vertex v, let N_K(v) = {u in K : (u,v) in edges}
        # Since K is a clique, any subset of N_K(v) union {v} forms a clique
        
        # Strategy: 
        # 1. The clique K itself covers all edges within K
        # 2. For cross edges, we need to cover each edge (u,v) where u in K, v in I
        # 3. Each such edge must be in exactly one clique
        # 4. A clique containing v and some subset S of N_K(v) covers all edges (u,v) for u in S
        # 5. We want to minimize total number of cliques
        
        # Key insight: The edge (u,v) for u in K, v in I can only be covered by a clique 
        # that contains both u and v. Since v is in I, the clique must be of the form {v} ∪ S 
        # where S ⊆ N_K(v).
        
        # Each such clique covers |S| cross edges. To minimize cliques, we want large S.
        # But we also need to ensure no edge is covered twice.
        
        # Since all I vertices are independent, no clique can contain two I vertices.
        # So each clique contains at most one I vertex.
        
        # The edges within K are all covered by the clique K itself (if we include it).
        # Wait, but if we include K as a clique, it covers all (k choose 2) edges within K.
        # Then for cross edges, we need separate cliques.
        
        # Alternative: Don't include K as a single clique. Instead, for each I vertex v,
        # create a clique {v} ∪ N_K(v). This covers all cross edges incident to v.
        # But then edges within K that are also in some {v} ∪ N_K(v) would be double-covered.
        
        # Let's think more carefully:
        # An edge (u,w) with u,w in K can be covered by:
        #   - A clique containing only K vertices
        #   - A clique {v} ∪ S where v in I and {u,w} ⊆ S ⊆ N_K(v)
        
        # To avoid double coverage, each edge must be in exactly one clique.
        
        # Simple valid solution:
        # 1. Include K as one clique (covers all K-K edges)
        # 2. For each v in I, include {v} ∪ N_K(v) as a clique (covers all cross edges incident to v)
        # This uses 1 + (n-k) cliques.
        # Total: 1 + n - k
        
        # Can we do better?
        # If N_K(v) is empty, we don't need a clique for v.
        # If N_K(v) has size 1, say {u}, then {v,u} is a 2-clique covering edge (u,v).
        
        # The K-K edges: if we use K as a clique, that's 1 clique for k(k-1)/2 edges.
        # Alternatively, we could cover K-K edges using the I-vertex cliques, but that would
        # require those I-vertex cliques to contain pairs of K vertices, which they do if
        # N_K(v) contains that pair.
        
        # Let's try a different approach:
        # For each v in I with N_K(v) non-empty, create clique {v} ∪ N_K(v).
        # This covers all cross edges.
        # Now, which K-K edges are covered? Edge (u,w) is covered if there exists v in I
        # such that {u,w} ⊆ N_K(v).
        
        # K-K edges not covered by any I-vertex clique must be covered separately.
        # We can cover them by adding cliques on subsets of K.
        
        # To minimize total cliques:
        # 1. For each v in I with |N_K(v)| >= 2, add clique {v} ∪ N_K(v)
        #    (covers all cross edges for v, and some K-K edges)
        # 2. For each v in I with |N_K(v)| == 1, add 2-clique {v, u} where u is the neighbor
        # 3. Cover remaining K-K edges with additional cliques on K
        
        # For step 3, the remaining K-K edges form a graph on K. We need a clique cover
        # of this graph. Since K is a clique, the remaining edges form a subgraph.
        # The minimum clique cover of a graph equals the maximum independent set size (by
        # complement duality), but for a subgraph of a clique, it's the chromatic number of
        # the complement, which equals the clique cover number.
        
        # Actually, for a graph G, min clique cover = max independent set of complement... 
        # Let me just use a greedy approach or note that if the remaining K-K graph is
        # dense, few cliques suffice.
        
        # Simpler: just use the naive solution first and optimize.
        
        cliques = []
        
        # Determine N_K(v) for each v in I
        N_K = {}
        for v in I:
            neighbors = set()
            for u in K:
                if (min(u, v), max(u, v)) in edges:
                    neighbors.add(u)
            N_K[v] = neighbors
        
        # Create cliques for I vertices
        covered_kk_edges = set()
        
        for v in I:
            if N_K[v]:
                clique = [v] + list(N_K[v])
                cliques.append(sorted(clique))
                # Mark all K-K edges within N_K[v] as covered
                for u, w in combinations(sorted(N_K[v]), 2):
                    covered_kk_edges.add((u, w))
        
        # Find remaining K-K edges
        remaining_kk = []
        for u, w in combinations(K, 2):
            if (u, w) not in covered_kk_edges:
                remaining_kk.append((u, w))
        
        # Cover remaining K-K edges
        # Greedy: repeatedly pick the largest clique that covers the most uncovered edges
        if remaining_kk:
            # Build adjacency for remaining K-K edges
            remaining_set = set(remaining_kk)
            
            # Greedy set cover-like approach for clique cover
            uncovered = set(remaining_kk)
            while uncovered:
                # Find a maximal clique in the remaining edge graph
                # Try each vertex as start
                best_clique = None
                best_count = 0
                for start in list(uncovered)[0]:
                    # Build candidate clique
                    cand = {start[0], start[1]}
                    # Try to expand
                    changed = True
                    while changed:
                        changed = False
                        for u in list(cand):
                            for w in K:
                                if w not in cand and (min(u, w), max(u, w)) in uncovered:
                                    # Check if w can be added (must connect to all in cand)
                                    can_add = True
                                    for x in cand:
                                        if x != w and (min(x, w), max(x, w)) not in uncovered and (min(x, w), max(x, w)) not in remaining_set:
                                            # This edge is not in remaining, so it's either covered or doesn't exist
                                            # For K-K edges, they all exist in the original graph
                                            # If not in remaining_set, it was already covered
                                            # We can still add w if all (x,w) for x in cand are in remaining or already covered
                                            # Actually, for the clique to be valid, all pairs must be edges
                                            # All K-K pairs are edges. The question is whether they're uncovered.
                                            # We want a clique that covers uncovered edges.
                                            pass
                                    # Simplified: just check if all pairs in cand ∪ {w} are edges (they are, since K is clique)
                                    # and at least one new uncovered edge is added
                                    new_edges = set()
                                    for x in cand:
                                        e = (min(x, w), max(x, w))
                                        if e in uncovered:
                                            new_edges.add(e)
                                    if new_edges:
                                        cand.add(w)
                                        changed = True
                                        break
                            if changed:
                                break
                    
                    if len(cand) > best_count:
                        best_count = len(cand)
                        best_clique = cand
                
                if best_clique is None or len(best_clique) < 2:
                    # Take the first uncovered edge
                    e = next(iter(uncovered))
                    best_clique = {e[0], e[1]}
                
                cliques.append(sorted(best_clique))
                # Remove covered edges
                for u, w in combinations(sorted(best_clique), 2):
                    e = (u, w)
                    if e in uncovered:
                        uncovered.discard(e)
        
        return cliques

    result = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        cliques = solve_split_graph(*spec)
        # Convert to list of lists of ints
        cliques = [list(c) for c in cliques]
        result.append(cliques)
    
    return result
# EVOLVE-BLOCK-END
