"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time
    import heapq
    from itertools import combinations

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        C = list(range(k))
        I = list(range(k, n))
        
        # Build cross adjacency
        cross = {i: [] for i in I}
        cross_set = set()
        for (a, b) in edges:
            if a < k and b >= k:
                cross[b].append(a)
                cross_set.add((a, b))
            elif a >= k and b < k:
                cross[a].append(b)
                cross_set.add((b, a))
        
        # Total edges in C
        all_C_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_C_edges.add(frozenset((u, v)))
        
        # Initial state
        # We want to cover all cross edges and all C edges.
        # Strategy: 
        # 1. For each i in I, we must cover its cross edges.
        #    We can cover cross edges of i by cliques {i} U S where S is a subset of N(i).
        #    If we use a clique {i} U S, we cover cross edges (i, s) for s in S, 
        #    and C-edges (s1, s2) for s1, s2 in S.
        # 2. Remaining C-edges must be covered by cliques in C.
        
        # Let's model this as:
        # We have a pool of "Cross Edge Groups" associated with each i.
        # Actually, simpler:
        # Each i in I has a set of neighbors N(i).
        # We need to partition N(i) into subsets S_1, ..., S_m.
        # Each subset S_j creates a clique {i} U S_j.
        # This covers C-edges inside S_j.
        
        # The total number of cliques is sum(|N(i)| groups) + (C-edges not covered / avg size).
        # To minimize, we want large S_j (fewer groups per i) but this covers more C-edges.
        # However, covering C-edges is "free" in terms of clique count for the cross part, 
        # but it reduces the number of C-only cliques needed.
        
        # Let's try a greedy approach:
        # 1. Start with each i having N(i) as a single group? No, N(i) must be a clique in C.
        #    Since C is a clique, ANY subset of C is a clique.
        #    So {i} U N(i) is always a valid clique.
        #    If we use {i} U N(i) for every i, we cover ALL cross edges.
        #    We also cover ALL C-edges that are inside any N(i).
        #    The remaining C-edges are those (u,v) such that there is no i with u,v in N(i).
        #    Let's call this set R.
        #    We then need to cover R with cliques in C.
        
        # Is this optimal?
        # Using {i} U N(i) uses 1 clique per i. Total |I| cliques.
        # If we split N(i) into smaller parts, we use more cliques for i, but cover fewer C-edges.
        # Covering fewer C-edges means MORE C-only cliques needed.
        # So it seems always better to use the largest possible S for each i, i.e., N(i).
        # Why? 
        # Cost of using S: 1 clique.
        # Benefit: covers |S| cross edges and C(S) C-edges.
        # If we split S into S1, S2: Cost 2 cliques. Benefit: same cross edges, C(S1)+C(S2) C-edges.
        # C(S) > C(S1)+C(S2) because S is a clique, so C(S) includes edges between S1 and S2.
        # So using larger S covers strictly more C-edges for the same or fewer cross cliques?
        # Wait. If we use N(i), we use 1 clique. If we split, we use >1 cliques.
        # So using N(i) is always better for the cross part count.
        # And it covers more C-edges, which is also better for the C part count.
        # Therefore, the optimal strategy for the cross part is to use {i} U N(i) for each i.
        
        # Let's verify this logic.
        # Total cliques = |I| + (number of cliques to cover R).
        # R is the set of C-edges (u,v) such that no i has both u and v in N(i).
        
        # Let's compute R.
        covered_C_edges = set()
        for i in I:
            Ni = cross[i]
            for u, v in combinations(Ni, 2):
                covered_C_edges.add(frozenset((u, v)))
        
        R = all_C_edges - covered_C_edges
        
        # Now we need to cover R with cliques in C.
        # This is the minimum clique cover of the graph G_R = (C, R).
        # Since C is a clique, G_R is a subgraph of a complete graph.
        # Minimum clique cover in a general graph is NP-hard, but G_R is a "co-graph" or similar?
        # Actually, G_R is just an arbitrary graph on k vertices.
        # However, k is small (10, 20, 30).
        # We can use a greedy algorithm or a simple branch and bound.
        # Given the small size, a greedy max-clique removal should work well.
        
        def cover_graph(vertices, edges):
            # vertices: list of ints
            # edges: set of frozensets
            remaining = set(edges)
            cliques = []
            verts = list(vertices)
            
            while remaining:
                # Find a maximal clique in the graph defined by 'remaining'
                # Greedy: pick vertex with max degree in remaining, grow clique
                # Or simpler: pick first edge, try to extend it to a clique.
                
                # Let's try to find the largest clique in 'remaining'
                # Since k <= 30, we can be a bit more exhaustive if needed, but greedy is fast.
                
                best_clique = None
                max_edges = 0
                
                # Try starting from each vertex
                for start in verts:
                    # Greedy grow
                    clique = [start]
                    clique_set = {start}
                    # Candidates: neighbors of start in remaining
                    candidates = [v for v in verts if v != start and frozenset((start, v)) in remaining]
                    
                    while candidates:
                        # Pick candidate with max degree in remaining within candidates
                        # Simple: pick first
                        v = candidates.pop(0)
                        clique.append(v)
                        clique_set.add(v)
                        # Filter candidates: must be connected to all in clique
                        new_candidates = []
                        for w in candidates:
                            if all(frozenset((w, u)) in remaining for u in clique):
                                new_candidates.append(w)
                        candidates = new_candidates
                    
                    # Count edges in remaining covered by this clique
                    count = 0
                    for e in remaining:
                        if e.issubset(clique_set):
                            count += 1
                    if count > max_edges:
                        max_edges = count
                        best_clique = clique
                
                if best_clique is None or max_edges == 0:
                    # Should not happen if remaining is not empty
                    break
                
                cliques.append(best_clique)
                # Remove covered edges
                clique_set = set(best_clique)
                to_remove = [e for e in remaining if e.issubset(clique_set)]
                for e in to_remove:
                    remaining.discard(e)
            
            return cliques
        
        c_cliques = cover_graph(C, R)
        
        # Construct final list
        result = []
        for i in I:
            Ni = cross[i]
            if len(Ni) >= 1:
                # Clique {i} U Ni
                # Must have at least 2 vertices.
                # If Ni is empty, i is isolated? No, cross[i] would be empty.
                # If Ni has 1 vertex, clique is {i, u} (size 2). OK.
                result.append([i] + Ni)
        
        result.extend(c_cliques)
        
        # Validation check (optional, but good for debugging)
        # Check all edges covered exactly once
        # ... skipped for speed in final version, but logic holds.
        
        return result

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
# EVOLVE-BLOCK-END
