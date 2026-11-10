def construct():
    from random import Random
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        C = list(range(k))
        I = list(range(k, n))
        
        # Identify cross edges and C edges
        cross_edges = set()
        C_edges = set()
        for (a, b) in edges:
            if a < k and b < k:
                C_edges.add(frozenset((a, b)))
            else:
                # Cross edge
                if a < k:
                    c, i = a, b
                else:
                    c, i = b, a
                cross_edges.add((i, c))
                
        # Greedy Set Cover for cross edges
        uncovered_cross = set(cross_edges)
        used_C_edges = set()
        cliques = []
        
        # Precompute neighbors in C for each i in I
        # neighbors[i] is a set of c in C such that (i,c) is an edge
        neighbors = {i: set() for i in I}
        for (i, c) in cross_edges:
            neighbors[i].add(c)
            
        while uncovered_cross:
            best_i = None
            best_S = set()
            best_count = 0
            
            for i in I:
                # S_i is the set of neighbors of i that are still uncovered
                S_i = set()
                for c in neighbors[i]:
                    if (i, c) in uncovered_cross:
                        S_i.add(c)
                if len(S_i) > best_count:
                    best_count = len(S_i)
                    best_i = i
                    best_S = S_i
                    
            if best_i is None:
                break # Should not happen if uncovered_cross is not empty
            
            # Add clique {best_i} U best_S
            clique = [best_i] + list(best_S)
            cliques.append(clique)
            
            # Update uncovered_cross
            for c in best_S:
                uncovered_cross.discard((best_i, c))
                
            # Update used_C_edges
            best_S_list = list(best_S)
            for idx_a in range(len(best_S_list)):
                for idx_b in range(idx_a + 1, len(best_S_list)):
                    used_C_edges.add(frozenset((best_S_list[idx_a], best_S_list[idx_b])))
                    
        # Cover remaining C edges
        remaining_C_edges = C_edges - used_C_edges
        
        # Greedy clique cover for remaining C edges
        # We want to partition remaining_C_edges into cliques
        # Heuristic: repeatedly pick the clique that covers the most remaining edges
        remaining = set(remaining_C_edges)
        C_list = list(C)
        
        while remaining:
            best_clique = None
            best_edge_count = 0
            
            # Try starting from each vertex in C
            for start in C_list:
                # Build a maximal clique containing start
                # We can use a simple greedy expansion
                clique = [start]
                clique_set = {start}
                
                # Candidates must be connected to all in clique via remaining edges
                # Note: The graph is defined by 'remaining' edges.
                # We need a clique in this graph.
                
                # To speed up, we can just try to add vertices that are connected to all current clique members
                # Since k is small, we can just check all v
                candidates = [v for v in C_list if v != start]
                
                # Sort candidates by degree in the remaining subgraph induced by candidates + clique?
                # Or just iterate.
                # Let's do a simple greedy: pick the vertex with the most edges to the current clique?
                # Actually, for a clique, the vertex must be connected to ALL in the clique.
                
                # Let's just find a maximal clique by iterating through vertices
                # This is O(k^2) per start, O(k^3) total per iteration.
                # k=30, so 27000 ops. Fine.
                
                # We'll try to build a clique greedily
                current_clique = [start]
                current_set = {start}
                
                # To make it better, we can pick the next vertex that is connected to all in current_set
                # and has the highest degree in the remaining graph?
                # Let's just pick the first one that works to keep it simple, or the one with max degree.
                
                # Let's try to be a bit smarter:
                # At each step, among candidates connected to all in current_clique, pick the one with max degree in the remaining graph.
                
                changed = True
                while changed:
                    changed = False
                    best_candidate = None
                    best_deg = -1
                    
                    for v in C_list:
                        if v in current_set:
                            continue
                        # Check if v is connected to all in current_set via remaining edges
                        is_connected = True
                        for u in current_set:
                            if u == v: continue
                            if frozenset((u, v)) not in remaining:
                                is_connected = False
                                break
                        if not is_connected:
                            continue
                        
                        # Count degree of v in the remaining graph
                        deg = 0
                        for u in C_list:
                            if u != v and frozenset((u, v)) in remaining:
                                deg += 1
                                
                        if deg > best_deg:
                            best_deg = deg
                            best_candidate = v
                            
                    if best_candidate is not None:
                        current_clique.append(best_candidate)
                        current_set.add(best_candidate)
                        changed = True
                        # Note: We don't remove edges yet, we just build the clique.
                        # The edges are "covered" when we finalize the clique.
                        
                # Calculate how many edges this clique covers
                edge_count = 0
                clique_set_final = set(current_clique)
                for e in remaining:
                    if e.issubset(clique_set_final):
                        edge_count += 1
                            
                if edge_count > best_edge_count:
                    best_edge_count = edge_count
                    best_clique = list(current_clique)
                    
            if best_clique is None or best_edge_count == 0:
                # Should not happen if remaining is not empty
                break
                
            cliques.append(best_clique)
            
            # Remove edges covered by best_clique
            best_clique_set = set(best_clique)
            for e in list(remaining):
                if e.issubset(best_clique_set):
                    remaining.discard(e)
                    
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
