"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from itertools import combinations
    
    deadline = time.time() + 145.0  # 5 second safety margin
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = set()
        # Clique on first k vertices
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges with probability p
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency for the graph
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Split graph structure:
        # K = vertices 0..k-1 (clique)
        # I = vertices k..n-1 (independent set)
        
        # Calculate S_v = N_K(v) for each v in I
        S_v = {}
        for v in range(k, n):
            S_v[v] = adj[v].intersection(set(range(k)))
            
        # Build conflict graph on I
        # v and w conflict if |S_v ∩ S_w| >= 2
        # We want to select a set of active v's (independent set in conflict graph)
        # to form large cliques C_v = {v} U S_v
        
        active_v = []
        used_v = set()
        
        # Greedy selection of active v's
        # Sort by size of S_v descending to pick large cliques first
        i_vertices = sorted(range(k, n), key=lambda x: -len(S_v[x]))
        
        for v in i_vertices:
            if v in used_v:
                continue
            # Check conflict with already selected active v's
            is_conflict = False
            for w in active_v:
                if len(S_v[v].intersection(S_v[w])) >= 2:
                    is_conflict = True
                    break
            if not is_conflict:
                active_v.append(v)
                used_v.add(v)
                
        # Initialize solution
        solution = []
        
        # Track covered edges in K
        # Edges in K are pairs (u, w) with u, w < k
        covered_K_edges = set()
        
        # 1. Add cliques for active v's
        for v in active_v:
            clique = [v] + list(S_v[v])
            if len(clique) >= 2:
                solution.append(clique)
                # Mark all K-edges in S_v[v] as covered
                sv = S_v[v]
                for u, w in combinations(sv, 2):
                    e = (min(u, w), max(u, w))
                    covered_K_edges.add(e)
                    
        # 2. Handle inactive v's
        # For each inactive v, edges (v, u) for u in S_v[v] must be covered.
        # We use cliques {v, u}.
        for v in range(k, n):
            if v in used_v:
                continue
            for u in S_v[v]:
                # Clique {v, u}
                # Check if edge (v, u) is already covered?
                # It shouldn't be, because only active v's cover cross edges,
                # and this v is inactive.
                # Also, no other inactive v' covers (v, u) because v is unique.
                solution.append([v, u])
                
        # 3. Cover remaining K-edges
        # Find all K-edges not in covered_K_edges
        remaining_K_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                e = (u, w)
                if e in edges and e not in covered_K_edges:
                    remaining_K_edges.add(e)
                    
        # Greedy partition of remaining_K_edges into cliques
        # While remaining_K_edges is not empty:
        #   Find a maximal clique in the graph formed by remaining_K_edges
        #   Add it to solution
        #   Remove its edges from remaining_K_edges
        
        # To find maximal clique in remaining edges:
        # The vertices are K. The edges are remaining_K_edges.
        # We can use Bron-Kerbosch or just greedy.
        # Since k is small (<=30), we can be simple.
        
        while remaining_K_edges:
            if time.time() > deadline - 5:
                # Fallback: add each edge as a 2-clique
                for e in remaining_K_edges:
                    solution.append([e[0], e[1]])
                break
                
            # Find a clique that covers the most edges
            # Simple heuristic: pick the vertex with highest degree in remaining graph,
            # then find its neighborhood, etc.
            # Or just pick the first edge, then expand.
            
            # Let's use a simple greedy:
            # Pick an edge (u, v)
            # Try to extend it to a maximal clique in the remaining graph
            # 
            # Better: Find the maximal clique with the most edges.
            # Since k is small, we can iterate over all subsets? No.
            # We can use Bron-Kerbosch on the remaining graph.
            
            # Build adjacency for remaining graph
            rem_adj = {u: set() for u in range(k)}
            for u, v in remaining_K_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
                
            # Find all maximal cliques in rem_graph
            maximal_cliques = []
            
            def bron_kerbosch(R, P, X):
                if not P and not X:
                    if len(R) >= 2:
                        maximal_cliques.append(frozenset(R))
                    return
                if not P:
                    return
                # Pivot
                U = P | X
                u = max(U, key=lambda x: len(P & rem_adj[x]))
                for v in list(P - rem_adj[u]):
                    bron_kerbosch(R | {v}, P & rem_adj[v], X & rem_adj[v])
                    P.remove(v)
                    X.add(v)
                    
            bron_kerbosch(frozenset(), frozenset(range(k)), frozenset())
            
            if not maximal_cliques:
                # Should not happen if there are edges
                for e in remaining_K_edges:
                    solution.append([e[0], e[1]])
                break
                
            # Pick the clique with the most edges
            # Edge count for clique C is binom(|C|, 2) but only those in remaining_K_edges
            # Since C is a clique in rem_graph, all its edges are in remaining_K_edges
            best_clique = max(maximal_cliques, key=lambda c: len(c))
            
            # Add to solution
            solution.append(list(best_clique))
            
            # Remove its edges from remaining_K_edges
            for u, v in combinations(best_clique, 2):
                e = (min(u, v), max(u, v))
                remaining_K_edges.discard(e)
                
        return solution
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 10:
            # Not enough time; return what we have plus empty lists for remaining
            while len(results) < 3:
                results.append([])
            break
        results.append(solve_one(*spec))
    
    while len(results) < 3:
        results.append([])
    
    return results
# EVOLVE-BLOCK-END
