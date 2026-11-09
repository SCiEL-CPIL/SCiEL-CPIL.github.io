"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random

    start_time = time.time()
    deadline = 110.0  # safety margin under 120s limit

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges_cc = []
        for u in range(k):
            for v in range(u + 1, k):
                edges_cc.append((u, v))

        edges_ci = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges_ci.append((u, v))

        return edges_cc, edges_ci

    def solve_split(n, k, p, seed):
        """
        For a split graph with clique C = {0..k-1} and independent set I = {k..n-1}:
        - C-C edges: covered by partition of C into pairs (each pair is a 2-clique)
          plus one remaining vertex if k is odd (handled by adding it to an existing pair
          or as a separate 2-clique with a neighbor).
        - C-I edges: for each v in I, group its neighbors in C into maximal cliques
          within the induced subgraph on N(v) ∩ C. Since C is a clique, any subset
          of C is a clique, so the maximal cliques in C ∩ N(v) are just {N(v) ∩ C}
          itself... wait, no. The cross-cliques must be cliques in the full graph.
          
        Actually, for C-I edges: each edge (u,v) with u in C, v in I must be in
        exactly one clique. A clique containing v and some subset S of C must have
        S ∪ {v} form a clique. Since I is independent, v can only be with C vertices.
        Since C is a clique, any subset of C plus v forms a clique iff all those C
        vertices are adjacent to v (which they are, since they're in N(v)).
        
        So for each v in I, we can put all of N(v) ∩ C together with v into ONE clique.
        But that would mean the C vertices in N(v) appear in both their C-C partition
        cliques AND this cross clique, which violates "every edge lies in exactly one
        listed clique."
        
        The constraint is: every EDGE must lie in exactly one listed clique. A vertex
        can appear in multiple cliques. So if vertex u is in clique A (covering C-C edges)
        and also in clique B (covering C-I edges), that's fine as long as no edge is
        double-covered.
        
        Let me reconsider: 
        - C-C edges: partition C into pairs. Each pair {u1, u2} is a clique covering edge (u1,u2).
          If k is odd, one vertex is left over; pair it with any other C vertex, but that
          edge would already be covered. Instead, we need a partition of the C-C edge set.
          
        Actually, re-reading: "every edge must lie in exactly one listed clique." This means
        we need an edge clique cover where each edge is in exactly one clique.
        
        For C (a k-clique): we need to partition the edges of K_k into cliques. The minimum
        number of cliques to partition K_k's edges is ceil(k/2) if we use pairs, but that
        only works if k is even. If k is odd, we need more.
        
        Actually, for K_k, the edge partition into cliques: each clique of size s covers C(s,2)
        edges. We want to minimize the number of cliques.
        
        For k=10: K_10 has 45 edges. Using 5 pairs: 5 cliques cover 5 edges each = 25 edges. Not enough.
        Wait, I'm confusing myself. A pair {u,v} is a 2-vertex clique covering exactly 1 edge.
        So to cover all C(k,2) edges with 2-cliques, we need C(k,2) cliques. That's what the
        seed does for C-C... no wait, the seed puts ALL of C as one clique.
        
        Seed: cliques = [list(range(k))] + [[u,v] for each C-I edge]
        This covers C-C edges all in one clique, and each C-I edge in its own 2-clique.
        Total = 1 + |E_ci|
        
        Can we do better? If we use larger cliques for C-C edges... but any larger clique
        within C would cover multiple C-C edges, but those edges can't be in any other clique.
        That's fine! The constraint is each edge in EXACTLY one clique, not that each vertex
        is in exactly one clique.
        
        So for C, we can use any edge partition of K_k. The minimum number of cliques to
        partition the edges of K_k: this is equivalent to finding a decomposition of K_k
        into complete graphs. By a theorem, K_k can be decomposed into K_2's (i.e., pairs)
        using C(k,2) cliques, or into larger cliques.
        
        Actually, the minimum number of cliques in an edge-partition of K_k is:
        - If k is even: k/2 (using a 1-factorization, but that gives matchings, not cliques)
        
        Wait, I think I'm overcomplicating. Let me just use the seed approach for C-C
        (one big clique) and focus on reducing the C-I edge count.
        
        For C-I: currently each edge is a separate 2-clique. Can we group multiple C-I
        edges into one clique? A clique containing v (in I) and vertices {u1, u2, ...} in C
        covers all edges (ui, v) for i in the set. This is valid if {u1,...,um, v} forms a
        clique, which it does since C is a clique and ui are all adjacent to v.
        
        So for each v in I, we can put ALL of N(v) ∩ C into a single clique with v.
        This covers all C-I edges incident to v in one clique.
        
        Total cliques = 1 (for C) + |I| = 1 + (n-k)
        
        For (30,10): 1 + 20 = 21, vs seed's 1 + |E_ci|. |E_ci| ≈ 0.7*10*20 = 140. So 21 << 140.
        For (60,20): 1 + 40 = 41, vs seed's ~1 + 0.7*20*40 = 561.
        For (90,30): 1 + 60 = 61, vs seed's ~1 + 0.7*30*60 = 1261.
        
        Total = 21 + 41 + 61 = 123, vs seed's ~1 + 140 + 561 + 1261 ≈ 1964.
        
        But wait: does this violate the "each edge in exactly one clique" constraint?
        - C-C edges: all in the one C clique. ✓
        - C-I edges: each edge (u,v) with u in C, v in I is in exactly the clique {N(v)∩C, v}. ✓
        - I-I edges: none exist (I is independent). ✓
        
        This is valid! And much better.
        
        But can we do even better? Can we group multiple I vertices into one clique?
        No, because I is independent, so two I vertices can't be in the same clique
        (unless the clique has only I vertices, but then it's not a clique since there's
        no edge between them... actually a clique of size >= 2 needs all pairs to be edges).
        So I vertices can't share a clique.
        
        Can we reduce the C clique? If we partition C into smaller cliques, each C-I
        edge still needs to be covered. But now a C vertex u in a small C-clique A might
        also need to be in a cross-clique for v. That's fine. But the C-C edges: if we
        split C into multiple cliques, we need those cliques to partition the C-C edges.
        
        Hmm, but using one big C clique is already minimal for covering all C-C edges
        in one clique. Splitting would require MORE cliques for C-C.
        
        So 1 + (n-k) seems optimal for this structure.
        
        Let me verify with the actual generator and count.
        """
        if time.time() > deadline - 5:
            # Fallback: seed-like
            edges_cc, edges_ci = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for u, v in edges_ci:
                cliques.append([u, v])
            return cliques

        edges_cc, edges_ci = instance(n, k, p, seed)

        cliques = []
        # One clique for all of C
        cliques.append(list(range(k)))

        # For each v in I, one clique with N(v) ∩ C
        # Build N(v) for each v in I
        N_in_C = [set() for _ in range(n)]
        for u, v in edges_ci:
            N_in_C[v].add(u)

        for v in range(k, n):
            neighbors = sorted(N_in_C[v])
            if len(neighbors) >= 1:
                cliques.append(neighbors + [v])

        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))

    for spec in specs:
        n, k, p, seed = spec
        cliques = solve_split(n, k, p, seed)
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
