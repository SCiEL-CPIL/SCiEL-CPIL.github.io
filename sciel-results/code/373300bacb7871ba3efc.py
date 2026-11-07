"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        # Generate edges
        # K is 0..k-1, I is k..n-1
        edges = set()
        # Edges in K
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Edges between K and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Precompute neighbors in K for each v in I
        # And neighbors in I for each u in K (not needed for this strategy)
        
        # Map edge (u,v) with u<v to a canonical form
        def canon(u, v):
            return (u, v) if u < v else (v, u)
        
        # Identify K and I vertices
        K = list(range(k))
        I = list(range(k, n))
        
        cliques = []
        covered_K_edges = set()
        
        # Process each vertex in I
        for v in I:
            # Find neighbors of v in K
            S_v = []
            for u in K:
                if canon(u, v) in edges:
                    S_v.append(u)
            
            if len(S_v) >= 2:
                # Form clique {v} U S_v
                clique = [v] + S_v
                cliques.append(clique)
                
                # Mark all edges within S_v as covered
                # S_v is a subset of K, so all pairs are edges in the graph
                for i in range(len(S_v)):
                    for j in range(i + 1, len(S_v)):
                        u1 = S_v[i]
                        u2 = S_v[j]
                        covered_K_edges.add(canon(u1, u2))
            
            elif len(S_v) == 1:
                u = S_v[0]
                clique = [v, u]
                cliques.append(clique)
                # No K-edges covered by a size-2 clique {v, u}
            
            # If len(S_v) == 0, no edges incident to v, do nothing
        
        # Cover remaining edges in K
        # Iterate all pairs in K
        for i in range(len(K)):
            for j in range(i + 1, len(K)):
                u1 = K[i]
                u2 = K[j]
                edge_key = canon(u1, u2)
                if edge_key not in covered_K_edges:
                    # This edge is not covered by any C_v
                    # Add a size-2 clique
                    cliques.append([u1, u2])
        
        # Ensure types are correct
        # Convert tuples to lists if any (we used lists)
        # Ensure ints are Python ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
            
        return final_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        results.append(solve_instance(*spec))
        
    return results
# EVOLVE-BLOCK-END
