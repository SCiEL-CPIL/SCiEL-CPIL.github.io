"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from itertools import combinations
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = []
        # C-C edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)
    
    def solve_split(n, k, p, seed):
        # Generate the split graph
        rng = Random(seed)
        # Build adjacency for cross edges
        S = [set() for _ in range(n)]  # S[i] = set of C-vertices connected to i
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    S[v].add(u)
        
        # I vertices are k..n-1
        I_vertices = list(range(k, n))
        
        # We need to cover:
        # 1. All C-C edges: k(k-1)/2 edges
        # 2. All cross edges: sum of |S[v]| for v in I
        #
        # Strategy:
        # - Pair up I vertices to form 4-vertex cliques (2C+2I) where possible
        # - Use 3-vertex cliques (1C+2I) or 2-vertex (1C+1I) for remaining
        # - Cover C-C edges with cliques within C
        #
        # For C-C edges: we can cover them with cliques of size 2 (edges) or larger
        # But each C-C edge must be in exactly one clique.
        # A clique within C of size m covers m(m-1)/2 C-C edges.
        # To minimize number of cliques for C-C edges, use large cliques.
        # But we also need to use C-vertices in cross-edge cliques.
        #
        # Key insight: C-vertices can appear in multiple cliques (one for C-C edges, 
        # and one or more for cross edges). The constraint is each EDGE is in exactly one clique.
        #
        # So:
        # - C-C edges: partition the edge set of K_k into cliques
        # - Cross edges: partition into cliques with at most 2 I-vertices each
        #
        # For C-C edges: minimum clique cover of K_k.
        # K_k can be covered by cliques. The minimum number of cliques to cover all edges
        # of K_k is achieved by using as large cliques as possible.
        # Actually, we can use a single clique of size k to cover ALL C-C edges!
        # But wait - that clique would be {0,1,...,k-1}, which is valid.
        # However, we also need to use C-vertices in cross-edge cliques.
        # The C-C edge clique and cross-edge cliques are separate - they share vertices but not edges.
        # So we can have {0,1,...,k-1} as one clique covering all C-C edges,
        # AND have cliques like {c, i} or {c1,c2,i,j} for cross edges.
        # These don't conflict because C-C edges are only in the first clique,
        # and cross edges are only in the second set.
        
        # So: 1 clique for all C-C edges: {0, 1, ..., k-1}
        cliques = [list(range(k))]
        
        # Now cover cross edges
        # For each i in I, S[i] is the set of C-vertices connected to i
        # We need to partition all edges (c,i) for c in S[i], i in I
        # into cliques where each clique has at most 2 I-vertices
        
        # Strategy: greedily pair I vertices
        # For pair (i,j), the clique can include C-vertices from S[i] ∩ S[j]
        # But we could also split: some C-vertices go to a clique with i only, some with j only,
        # and common ones go to a pair clique.
        
        # Better: for each i, we need to cover all edges in S[i].
        # If we pair i with j, we can put T ⊆ S[i] ∩ S[j] in a clique {T, i, j},
        # and the rest S[i] \ T in cliques with i alone, S[j] \ T in cliques with j alone.
        
        # To minimize total cliques, we want to maximize the number of edges covered by pair-cliques.
        # A pair-clique {T, i, j} with |T| = t covers 2t edges with 1 clique.
        # Individual cliques: each {c, i} covers 1 edge with 1 clique.
        # So a pair-clique saves (2t - 1) cliques compared to individual, for t >= 1.
        
        # Greedy: sort pairs by |S[i] ∩ S[j]| descending, greedily match.
        
        # Build intersection sizes
        m = len(I_vertices)
        # Create index mapping
        idx = {v: i for i, v in enumerate(I_vertices)}
        
        # Compute intersection sizes for all pairs
        pairs = []
        for a in range(m):
            for b in range(a + 1, m):
                ia = I_vertices[a]
                ib = I_vertices[b]
                inter = len(S[ia] & S[ib])
                if inter > 0:
                    pairs.append((-inter, a, b))
        
        # Sort by intersection size descending
        pairs.sort()
        
        # Greedy matching
        matched = [False] * m
        pairings = []
        for neg_inter, a, b in pairs:
            if not matched[a] and not matched[b]:
                matched[a] = True
                matched[b] = True
                pairings.append((a, b))
        
        # For each pair, create clique with common C-vertices
        for a, b in pairings:
            ia = I_vertices[a]
            ib = I_vertices[b]
            common = S[ia] & S[ib]
            if common:
                clique = sorted(common) + [ia, ib]
                cliques.append(clique)
                # Remove these edges from S
                S[ia] -= common
                S[ib] -= common
        
        # For remaining I vertices (including those that were paired but have leftover edges,
        # and those that weren't paired), cover remaining edges
        for i in I_vertices:
            remaining = S[i]
            if remaining:
                # We can create one clique per C-vertex, or group them
                # A clique {c1, c2, ..., cm, i} is valid if all c's connect to i
                # Since remaining ⊆ S[i], all connect to i, and C is a clique,
                # so {remaining, i} is a valid clique!
                clique = sorted(remaining) + [i]
                if len(clique) >= 2:
                    cliques.append(clique)
                S[i] = set()
        
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_split(n, k, p, seed)
        # Convert to proper types
        cliques = [list(c) for c in cliques]
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
