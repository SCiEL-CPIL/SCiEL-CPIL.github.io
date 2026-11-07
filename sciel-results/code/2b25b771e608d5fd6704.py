def construct():
    from random import Random
    import time
    import itertools

    start_time = time.time()
    DEATHLINE = 145  # leave margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def split_graph_clique_partition(n, k, p, seed):
        """
        Structural approach for split graphs:
        - C = {0,...,k-1} (clique)
        - I = {k,...,n-1} (independent set)
        
        Edges: C-C (all), C-I (with prob p), I-I (none).
        
        Strategy:
        1. Use C as one clique to cover all C-C edges.
        2. For I-C edges, we need to cover them with cliques that include vertices from both C and I.
           Each such clique can be {v} ∪ S where v ∈ I and S ⊆ C, but S must be a subset of N(v) (neighbors of v in C).
           The clique {v} ∪ S covers edges (v, s) for s ∈ S, and edges (s1, s2) for s1,s2 ∈ S (but those are already covered by C).
           Wait - we can't re-cover C-C edges. So each I-C clique should be exactly {v} ∪ {c} (an edge) OR we can use larger cliques 
           but they would re-cover C-C edges.
           
        Actually, the constraint is: every edge must lie in EXACTLY ONE listed clique. So we cannot have overlapping cliques that share edges.
        
        Since C is a clique, if we use C as a clique, then no other clique can contain two vertices from C (that would create an overlapping edge).
        
        So for I-C edges, each clique can contain at most one vertex from C and any number from I? No, because I is independent, so a clique with multiple I vertices and one C vertex would need those I vertices to be connected to each other, which they're not.
        
        Wait: a clique {v1, v2, c} where v1,v2 ∈ I and c ∈ C: we need v1-v2 edge (no, I is independent), v1-c edge (yes), v2-c edge (yes). So this is NOT a clique because v1-v2 is not an edge.
        
        So any clique containing vertices from both C and I can contain at most ONE vertex from I (since I is independent, two I vertices can't be in the same clique) and any number from C? No wait:
        
        A clique {v, c1, c2, c3} where v ∈ I and c1,c2,c3 ∈ C: need v-ci edges for all ci (yes if ci ∈ N(v)), and ci-cj edges for all ci,cj ∈ C (yes, C is a clique). So this IS a valid clique!
        
        But if we use C as a clique, then the edge (c1,c2) is already covered by C. If we also put c1,c2 in {v,c1,c2,c3}, then edge (c1,c2) is in two cliques - violation!
        
        So we cannot use C as a clique AND also use cliques with multiple C vertices.
        
        Alternative: don't use C as a single clique. Instead, cover C-C edges and I-C edges together.
        
        Let me think differently. The key constraint is edge-disjointness.
        
        Approach 1: Use C as one clique. Then for I-C edges, each clique is {v, c} (single edge) because we can't have two C vertices in another clique. This gives k*(n-k)*p edges, each as a 2-clique. Plus one clique for C. Total: 1 + |I-C edges|.
        
        Approach 2: Don't use C as a single clique. For each v ∈ I, create clique {v} ∪ N(v). This covers all I-C edges incident to v, and also covers C-C edges within N(v). But then C-C edges not in any N(v) need separate coverage, and C-C edges in multiple N(v) would be double-covered.
        
        This is tricky. Let me try a different structural approach.
        
        Since the graph is a split graph, the chromatic number is k (size of clique C). The clique cover number... 
        
        Let me try: for each v ∈ I, let S_v = N(v) ⊆ C. Create clique {v} ∪ S_v. This covers all edges (v, s) for s ∈ S_v, and all edges (s1, s2) for s1,s2 ∈ S_v.
        
        The C-C edges that are covered are those in ∪_v S_v × S_v (edges within each S_v). The remaining C-C edges need to be covered by cliques that don't include any I vertices (since adding an I vertex w would require w to be connected to both endpoints, i.e., both in S_w, but that's a separate clique).
        
        Actually, any remaining C-C edge (a,b) where (a,b) is not in any S_v × S_v can be covered by a 2-clique {a,b}.
        
        But we also need to ensure no edge is covered twice. Edge (a,b) in C-C is covered by clique {v}∪S_v iff a,b ∈ S_v. So if (a,b) is in multiple S_v's, it's covered multiple times - bad.
        
        So this approach only works if the S_v's are edge-disjoint in terms of C-C edges. Which is generally not the case.
        
        Let me go back to the simple approach: 1 clique for C, and one 2-clique per I-C edge. Total: 1 + |E_IC|.
        
        For n=90, k=30, p=0.7: |E_IC| ≈ 30*60*0.7 = 1260. Total ≈ 1261.
        
        The theoretical bound is 3n²/16 + O(n) = 3*8100/16 ≈ 1518.75. So 1261 is within the bound.
        
        Can we do better? The Erdős-Ordman-Zalcstein conjecture suggests n²/6 + O(n) ≈ 1350. So we might be able to do around 1350.
        
        Let me try a greedy approach that's smarter:
        - Start with no cliques.
        - Repeatedly find the largest clique that covers the most uncovered edges.
        - But ensure edge-disjointness.
        
        For split graphs, the maximum cliques are either C itself, or {v} ∪ N(v) for v ∈ I.
        
        Let me try: 
        1. For each v ∈ I, compute N(v) ⊆ C.
        2. Use a greedy set cover: at each step, pick the clique (either C or {v}∪N(v)) that covers the most uncovered edges, add it to the partition, and mark those edges as covered.
        3. But we need to be careful: if we use {v}∪N(v), it covers edges (v,s) for s∈N(v) AND edges (s1,s2) for s1,s2∈N(v). The latter are C-C edges.
        
        The issue is that C-C edges might be covered by multiple {v}∪N(v) cliques. To handle this, we should prefer cliques that cover more unique edges.
        
        Let me implement a greedy that tracks uncovered edges and picks the best clique each time.
        """
        edges = instance(n, k, p, seed)
        edge_set = set((min(u,v), max(u,v)) for u, v in edges)
        
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        C = list(range(k))
        I = list(range(k, n))
        
        # Precompute N(v) for v in I
        Nv = {v: adj[v] & set(C) for v in I}
        
        uncovered = set(edge_set)
        cliques = []
        
        # Candidate cliques: C and {v} ∪ N(v) for each v in I
        candidate_cliques = [C]
        for v in I:
            if Nv[v]:
                candidate_cliques.append([v] + sorted(Nv[v]))
        
        # Greedy: at each step, pick the clique that covers the most uncovered edges
        while uncovered:
            if time.time() - start_time > DEATHLINE - 10:
                break
            
            best_clique = None
            best_count = 0
            
            for clique in candidate_cliques:
                count = 0
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (min(a,b), max(a,b))
                        if e in uncovered:
                            count += 1
                if count > best_count:
                    best_count = count
                    best_clique = clique
            
            if best_clique is None or best_count == 0:
                # No candidate clique covers any uncovered edge; use a 2-clique
                e = next(iter(uncovered))
                best_clique = sorted([e[0], e[1]])
            
            cliques.append(best_clique)
            for i in range(len(best_clique)):
                for j in range(i+1, len(best_clique)):
                    a, b = best_clique[i], best_clique[j]
                    e = (min(a,b), max(a,b))
                    uncovered.discard(e)
        
        # Cover remaining edges with 2-cliques
        for e in sorted(uncovered):
            cliques.append(sorted([e[0], e[1]]))
        
        return cliques

    def simple_partition(n, k, p, seed):
        """Fallback: 1 clique for C, one 2-clique per I-C edge."""
        edges = instance(n, k, p, seed)
        edge_set = set((min(u,v), max(u,v)) for u, v in edges)
        
        C = list(range(k))
        cross_edges = [(u, v) for (u, v) in edge_set if (u < k <= v) or (v < k <= u)]
        
        cliques = [C] + [sorted([u, v]) for u, v in cross_edges]
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        greedy_cliques = split_graph_clique_partition(n, k, p, seed)
        simple_cliques = simple_partition(n, k, p, seed)
        
        if len(greedy_cliques) <= len(simple_cliques):
            results.append(greedy_cliques)
        else:
            results.append(simple_cliques)
    
    # Convert to ensure proper types
    final = []
    for clique_list in results:
        final.append([list(int(v) for v in clique) for clique in clique_list])
    
    return final
