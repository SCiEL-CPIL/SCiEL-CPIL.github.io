"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random
    from collections import defaultdict

    start_time = time.time()
    deadline = 145.0  # 150s limit, keep margin

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

    def solve_split_graph(n, k, edges_ci):
        """
        Structural approach for split graph clique partition:
        1. The clique part C = {0, ..., k-1} is a clique.
        2. The independent set I = {k, ..., n-1} has no edges between them.
        3. Cross edges form a bipartite graph between C and I.
        
        Key insight: Any clique in the full graph that contains vertices from I
        can contain at most... wait, no. Vertices in I are not connected to each other,
        so a clique can contain at most ONE vertex from I.
        
        Therefore, every clique that covers a cross edge (u, v) where u in C, v in I
        must be of the form {u, v, ...} where all other vertices are in C and
        connected to both u and v. Since C is a clique, any subset of C is a clique.
        The constraint is: if we put v (from I) in a clique with a subset S of C,
        then v must be adjacent to ALL vertices in S.
        
        So the problem reduces to:
        - We have a bipartite graph with left side C and right side I.
        - We need to cover all edges with cliques.
        - A clique containing a vertex v in I can include v and any subset of N(v) in C.
        - A clique not containing any vertex from I is just a subset of C.
        
        Since C is a clique, we can cover all C-C edges with a single clique C itself.
        But then we must NOT include any cross edges in that clique (since cross edges
        connect C to I, and I vertices aren't in C).
        
        Wait, the clique C = {0, ..., k-1} only contains edges between vertices in C.
        Cross edges (u, v) with u in C, v in I are NOT in this clique.
        
        So we need separate cliques for the cross edges. Each such clique must contain
        exactly one vertex from I (since I is independent) and some vertices from C.
        
        For a vertex v in I with neighborhood N(v) in C, we need to cover all edges
        (u, v) for u in N(v). We can do this with a single clique {v} union S where
        S is a subset of N(v). But to minimize the number of cliques, we want each
        such clique to cover as many cross edges as possible.
        
        However, each clique can only cover edges incident to ONE vertex in I.
        So the minimum number of cliques needed for cross edges is at least |I| (one per
        vertex in I that has at least one edge), and we can achieve exactly |I| by
        using for each v in I the clique {v} union N(v).
        
        Wait, but {v} union N(v) is a clique iff v is adjacent to all vertices in N(v),
        which is true by definition of N(v). And N(v) is a subset of C, which is a clique.
        So yes, {v} union N(v) is a clique.
        
        But can we do better? Can we merge some of these?
        No, because each clique can contain at most one vertex from I. So we need at
        least one clique per vertex in I that has at least one cross edge.
        
        Total cliques = 1 (for C-C edges) + |{v in I : deg(v) > 0}|
        
        But wait, can we avoid the single clique for C? If we don't use C as a clique,
        we need to cover C-C edges with other cliques. Each such clique can contain
        at most one vertex from I, so it would be {v, S} where S is a subset of C and
        S subset N(v). The C-C edges within S are covered by this clique. But we'd need
        many such cliques to cover all C-C edges, which is worse than just using C.
        
        So the optimal is: 1 + number of vertices in I with degree > 0.
        
        Let me verify: 
        - n=30, k=10: I has 20 vertices. Each has probability 1-(1-0.7)^10 ≈ 0.99998 of
          having at least one edge. So ~20 vertices in I have edges. Total ≈ 21.
        - n=60, k=20: I has 40 vertices. Similar, ~40. Total ≈ 41.
        - n=90, k=30: I has 60 vertices. ~60. Total ≈ 61.
        
        But the current program scores 1.3357, meaning seed/value. If seed is some baseline,
        and value is total cliques... let me think about what the "seed" value is.
        
        Actually, the task says "the search score is seed/value, so 1.0 equals the seed program".
        So the seed program has some number of cliques, and we want to minimize the total.
        
        My analysis says the minimum is 1 + |{v in I : N(v) != empty}|.
        This is essentially optimal since we can't do better than one clique per I-vertex.
        
        Let me implement this.
        """
        cliques = []
        
        # 1. Big clique for all C-C edges
        if k >= 2:
            cliques.append(list(range(k)))
        
        # 2. For each vertex v in I with at least one cross edge, create clique {v} union N(v)
        # Build adjacency: for each v in I, find all u in C connected to v
        adj = defaultdict(list)
        for u, v in edges_ci:
            adj[v].append(u)
        
        for v in range(k, n):
            if adj[v]:
                clique = [v] + adj[v]
                if len(clique) >= 2:
                    cliques.append(clique)
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: individual edges
            n, k, p, seed = spec
            edges_cc, edges_ci = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for u, v in edges_ci:
                cliques.append([u, v])
            results.append(cliques)
            continue
            
        n, k, p, seed = spec
        edges_cc, edges_ci = instance(n, k, p, seed)
        
        cliques = solve_split_graph(n, k, edges_ci)
        
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
