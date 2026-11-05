def construct():
    from itertools import combinations
    import random

    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_clique_partition(n, edges):
        if not edges:
            return []
        
        remaining_edges = set(edges)
        cliques = []
        
        # Precompute full adjacency for quick lookup? 
        # No, we need adjacency in the REMAINING graph.
        # We will recompute degrees and adjacency for the remaining graph in each step.
        # n is small (32, 64), so this is feasible.
        
        while remaining_edges:
            # 1. Compute degrees in remaining graph
            deg = {v: 0 for v in range(n)}
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            # 2. Find a large clique
            # Strategy: Try to build a clique starting from several high-degree vertices
            # and pick the best one.
            
            # Get vertices sorted by degree
            sorted_verts = sorted(range(n), key=lambda x: deg[x], reverse=True)
            
            best_clique = []
            best_clique_edges_count = 0
            
            # Try starting from the top K vertices
            K = 10
            for start in sorted_verts[:K]:
                if deg[start] == 0:
                    continue
                
                # Build a clique greedily starting from start
                # Candidate set: neighbors of start in remaining graph
                # A neighbor v is valid if (start, v) is in remaining_edges
                candidates = set()
                for u, v in remaining_edges:
                    if u == start:
                        candidates.add(v)
                    elif v == start:
                        candidates.add(u)
                
                if not candidates:
                    continue
                
                clique = [start]
                current_candidates = candidates
                
                while current_candidates:
                    # Pick candidate with highest degree in the induced subgraph of current_candidates?
                    # Or just highest degree in the whole remaining graph?
                    # Let's use degree in the remaining graph restricted to current_candidates + clique
                    # Actually, simpler: pick candidate with max degree in remaining graph.
                    
                    # To be more effective, we should ensure the new vertex is connected to ALL in clique.
                    # Since we only keep candidates that are connected to start, and we update candidates
                    # to be intersection of neighbors of new vertex and old candidates, 
                    # the condition "connected to all in clique" is maintained if we update correctly.
                    
                    # Update candidates: must be adjacent to the new vertex
                    # And must be adjacent to all existing clique members (which they are, by construction of the intersection if we do it right).
                    
                    # Let's pick the vertex in current_candidates with the highest degree in the remaining graph.
                    best_v = max(current_candidates, key=lambda x: deg[x])
                    
                    # Add best_v to clique
                    clique.append(best_v)
                    
                    # Update current_candidates
                    # New candidates must be neighbors of best_v in remaining graph
                    # AND must be in current_candidates
                    new_candidates = set()
                    for u, v in remaining_edges:
                        if u == best_v:
                            if v in current_candidates:
                                new_candidates.add(v)
                        elif v == best_v:
                            if u in current_candidates:
                                new_candidates.add(u)
                    
                    current_candidates = new_candidates
                
                # Check if this clique is valid (all edges present)
                # By construction, it should be, but let's verify edge count
                clique_set = set(clique)
                edge_count = 0
                valid = True
                for u in clique_set:
                    for v in clique_set:
                        if u < v:
                            if (u, v) not in remaining_edges and (v, u) not in remaining_edges:
                                # We store edges as (min, max) usually? 
                                # My generate_edges uses (u, v) with u < v.
                                # So check (min, max)
                                if (min(u,v), max(u,v)) not in remaining_edges:
                                    valid = False
                                    break
                    if not valid:
                        break
                    edge_count += len(clique_set) - 1 # This logic is messy, just count pairs
                
                if valid:
                    # Count edges in this clique
                    c_edges = set()
                    for u, v in combinations(clique, 2):
                        c_edges.add((min(u,v), max(u,v)))
                    
                    # Ensure all these edges are in remaining_edges
                    if c_edges.issubset(remaining_edges):
                        if len(c_edges) > best_clique_edges_count:
                            best_clique_edges_count = len(c_edges)
                            best_clique = clique
            
            if not best_clique:
                # Fallback: take any edge
                e = next(iter(remaining_edges))
                cliques.append([e[0], e[1]])
                remaining_edges.remove(e)
            else:
                # Remove edges of best_clique
                for u, v in combinations(best_clique, 2):
                    e = (min(u,v), max(u,v))
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                cliques.append(sorted(best_clique))
        
        return cliques

    def solve_instance(n, p, seed):
        edges = generate_edges(n, p, seed)
        return greedy_clique_partition(n, edges)
    
    return [solve_instance(n, p, seed) for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602))]
