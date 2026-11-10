import sys
from random import Random

def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        # Generate edges
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        
        # Adjacency representation for original graph (for quick neighbor lookup if needed, but we use remaining_edges)
        # Actually, we just need to iterate remaining_edges.
        
        remaining_edges = set(edges)
        cliques = []
        
        # Helper to find max clique in a subgraph defined by a set of vertices and a local adjacency
        def find_max_clique(vertices, local_adj):
            if not vertices:
                return []
            
            v_list = list(vertices)
            
            # Sort by degree in the subgraph
            def degree_in_sub(v):
                return len(local_adj.get(v, set()))
                
            v_list.sort(key=degree_in_sub, reverse=True)
            
            best = []
            
            def backtrack(current, candidates):
                nonlocal best
                if len(current) > len(best):
                    best = list(current)
                if not candidates:
                    return
                
                # Pruning
                if len(current) + len(candidates) <= len(best):
                    return

                for i in range(len(candidates)):
                    v = candidates[i]
                    new_current = current + [v]
                    # New candidates must be in candidates[i+1:] and connected to v in local_adj
                    new_candidates = [u for u in candidates[i+1:] if u in local_adj.get(v, set())]
                    backtrack(new_current, new_candidates)
                    
            backtrack([], v_list)
            return best

        while remaining_edges:
            # Compute degrees in remaining graph
            deg = {i: 0 for i in range(n)}
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
                
            # Get top K vertices by degree
            # Only consider vertices with degree > 0
            active_vertices = [v for v in range(n) if deg[v] > 0]
            if not active_vertices:
                break
                
            top_k = sorted(active_vertices, key=lambda x: deg[x], reverse=True)[:5]
            
            best_clique = []
            
            for v in top_k:
                # Neighbors of v in remaining graph
                neighbors = set()
                for u, w in remaining_edges:
                    if u == v:
                        neighbors.add(w)
                    elif w == v:
                        neighbors.add(u)
                        
                if not neighbors:
                    continue
                
                # Build local adjacency for neighbors
                local_adj = {u: set() for u in neighbors}
                # Iterate through remaining edges to build local_adj
                # Optimization: Only check edges where both endpoints are in neighbors
                # This is O(E) per candidate. E is shrinking.
                for u, w in remaining_edges:
                    if u in neighbors and w in neighbors:
                        local_adj[u].add(w)
                        local_adj[w].add(u)
                        
                clique_in_neighbors = find_max_clique(neighbors, local_adj)
                candidate = clique_in_neighbors + [v]
                
                if len(candidate) > len(best_clique):
                    best_clique = candidate
            
            if not best_clique:
                # Fallback
                e = next(iter(remaining_edges))
                best_clique = [e[0], e[1]]
            
            # Remove edges of best_clique
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    u, v = best_clique[i], best_clique[j]
                    if u > v:
                        u, v = v, u
                    remaining_edges.discard((u, v))
            
            cliques.append(best_clique)
            
        return cliques

    res1 = instance(32, 0.6, 601)
    res2 = instance(64, 0.55, 602)
    
    return [res1, res2]
