"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time
    
    start_time = time.time()
    deadline = 120.0 # 120s limit, keep margin
    
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

    def solve_instance(n, k, p, seed):
        # Generate the graph
        edges_cc, edges_ci = instance(n, k, p, seed)
        
        # Build adjacency sets for efficiency
        adj = [set() for _ in range(n)]
        for u, v in edges_cc:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in edges_ci:
            adj[u].add(v)
            adj[v].add(u)
            
        # Track remaining edges to cover
        # We need to cover all edges. 
        # Edges within C (0..k-1) must be covered.
        # Edges between C and I (k..n-1) must be covered.
        
        # Strategy:
        # 1. Find maximal cliques. 
        # 2. Since it's a split graph, any clique is either:
        #    - A subset of C
        #    - A subset of C plus a single vertex from I (since I is independent)
        #    - A single vertex from I (not a clique of size >= 2, so irrelevant unless part of larger)
        #    Actually, a clique can contain at most one vertex from I.
        #    So any clique K is of the form {v} U S where v in I, S subset of N(v) ∩ C.
        #    Or K is a subset of C.
        
        # We want to cover all edges with minimum number of cliques.
        # This is equivalent to Edge Clique Cover.
        
        # Let's use a greedy approach:
        # While there are uncovered edges:
        #   Find the largest clique that covers the most uncovered edges.
        #   Add it to the cover.
        #   Remove its edges from the uncovered set.
        
        # To make it efficient, we can precompute potential cliques.
        # Potential cliques involving I vertices:
        # For each v in I, let S_v = N(v) ∩ C. The clique is {v} ∪ S_v.
        # Potential cliques within C:
        # Any subset of C is a clique. The largest is C itself.
        
        # Let's analyze the structure:
        # Edges in C: All pairs (u, v) with u, v in C.
        # Edges between C and I: (u, v) with u in C, v in I, if edge exists.
        
        # If we pick the clique C (all k vertices), it covers all edges within C.
        # It does NOT cover any edges between C and I.
        
        # If we pick a clique {v} ∪ S_v for v in I, it covers:
        # - Edges (u, v) for u in S_v.
        # - Edges (u1, u2) for u1, u2 in S_v. (These are edges within C).
        
        # So, using cliques from I helps cover edges within C as well!
        
        # Let's maintain a count of how many times each edge is "needed".
        # Actually, we just need to cover each edge at least once.
        
        # Let's use a set of uncovered edges.
        uncovered_edges = set()
        for u, v in edges_cc:
            uncovered_edges.add((u, v))
        for u, v in edges_ci:
            uncovered_edges.add((u, v))
            
        cliques = []
        
        # Precompute the "candidate" cliques involving I vertices.
        # For each v in I, candidate clique is {v} U N(v) ∩ C.
        # Note: N(v) ∩ C is exactly the set of u in C such that (u, v) is an edge.
        
        candidates = []
        for v in range(k, n):
            neighbors_in_c = [u for u in range(k) if u in adj[v]]
            if len(neighbors_in_c) >= 1:
                clique = [v] + neighbors_in_c
                # We only consider cliques of size >= 2
                if len(clique) >= 2:
                    candidates.append(clique)
                    
        # Also, the clique C itself is a candidate.
        # But maybe we don't need to explicitly add C if the I-cliques cover all internal C edges?
        # Let's see. If we pick enough I-cliques, they cover edges within C.
        # Specifically, an edge (u, w) in C is covered by an I-clique {v} ∪ S_v if u, w in S_v.
        
        # Greedy selection:
        # At each step, find the candidate clique (from candidates or C) that covers the most uncovered edges.
        # To be efficient, we can iterate.
        
        # Let's add C to the pool of candidates, but we must be careful not to pick it if it's not optimal.
        # Actually, C is just one candidate.
        
        all_candidates = candidates + [list(range(k))]
        
        # Optimization: Remove candidates that are subsets of others?
        # If A is a subset of B, and both are cliques, B covers all edges A covers plus more.
        # So A is never better than B for covering edges, unless B is already used? No, we pick disjoint sets of edges?
        # No, edges can be covered multiple times, but we want to minimize the number of cliques.
        # So picking the largest clique is generally good.
        
        # However, a smaller clique might cover specific edges that the larger one doesn't?
        # No, if A is subset of B, edges(A) is subset of edges(B).
        # So B is strictly better or equal.
        # Therefore, we can discard any candidate that is a proper subset of another candidate.
        
        # Let's filter candidates.
        # This might be expensive if done naively O(N^2).
        # N is number of candidates.
        # Candidates are:
        # 1. {v} U N(v) for v in I.
        # 2. C.
        
        # Note: {v} U N(v) contains v. C does not contain any v in I.
        # So no candidate is a subset of C (unless N(v) is empty, but then size 1, excluded).
        # Can {v} U N(v) be a subset of {w} U N(w)?
        # Only if v = w (since v is unique to that clique).
        # So the I-cliques are all distinct and not subsets of each other.
        # Can C be a subset of an I-clique? No, C has k vertices, I-clique has 1 + |N(v)| vertices.
        # Wait, if N(v) = C, then I-clique is {v} U C. This contains C.
        # So if there is a vertex v in I connected to ALL vertices in C, then C is a subset of {v} U C.
        # In that case, {v} U C is better than C.
        
        # So we can just use the greedy algorithm on the list of candidates.
        # But we need to be careful: if we pick {v} U C, it covers all edges in C.
        
        # Let's implement the greedy loop.
        
        while uncovered_edges:
            best_clique = None
            best_count = 0
            
            # Check all candidates
            # This is O(|Candidates| * |Edges|) which might be slow if we do it naively.
            # |Candidates| <= n. |Edges| <= n^2.
            # n=90, n^2=8100. n * n^2 = 729000. This is fine for one step.
            # But how many steps?
            # Each step removes at least 1 edge.
            # Total edges ~ n^2/2.
            # So ~ n^2 steps.
            # Total complexity ~ n^4. 90^4 = 65M. Might be too slow in Python?
            # Let's optimize.
            
            # Alternative:
            # 1. Pick the clique that covers the most uncovered edges.
            # 2. Remove those edges.
            
            # Let's try to be smarter.
            # The edges in C are "internal". The edges between C and I are "cross".
            
            # Let's just run the greedy loop with a time check.
            
            for clique in all_candidates:
                # Count uncovered edges in this clique
                count = 0
                # Iterate over pairs in clique
                m = len(clique)
                for i in range(m):
                    u = clique[i]
                    for j in range(i+1, m):
                        v = clique[j]
                        # Normalize edge
                        if u > v:
                            u, v = v, u
                        if (u, v) in uncovered_edges:
                            count += 1
                
                if count > best_count:
                    best_count = count
                    best_clique = clique
                    
            if best_clique is None or best_count == 0:
                # This should not happen if graph is connected or has edges
                # But if there are isolated vertices? They don't form edges.
                # If uncovered_edges is not empty, there must be a clique covering it.
                # Unless we ran out of candidates?
                # We have all maximal cliques (mostly).
                # Actually, is it possible that an edge is not in any candidate?
                # Every edge (u, v) is in the clique {u, v}.
                # Is {u, v} in all_candidates?
                # If u, v in C, it's in C.
                # If u in C, v in I, it's in {v} U N(v).
                # So yes.
                
                # If best_count == 0, it means no candidate covers any uncovered edge?
                # That implies uncovered_edges is empty?
                # Or there is a bug.
                break
            
            # Add best_clique to result
            cliques.append(best_clique)
            
            # Remove covered edges
            m = len(best_clique)
            for i in range(m):
                u = best_clique[i]
                for j in range(i+1, m):
                    v = best_clique[j]
                    if u > v:
                        u, v = v, u
                    uncovered_edges.discard((u, v))
                    
            # Time check
            if time.time() > deadline - 5:
                # Fallback: just add 2-cliques for remaining edges
                for u, v in uncovered_edges:
                    cliques.append([u, v])
                break
                
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
