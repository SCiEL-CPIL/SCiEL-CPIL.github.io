def construct():
    import random
    import sys

    def instance(n, p, seed):
        rng = random.Random(seed)
        # Build adjacency matrix
        adj = [[False] * n for _ in range(n)]
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = adj[v][u] = True
                    edges.add((u, v))
        return adj, edges

    def solve(n, adj, edges):
        remaining = set(edges)
        cliques = []
        
        # Helper to check if a set is a clique in the original graph
        # (We assume we only add valid cliques, so this is for safety if needed, 
        # but the greedy construction ensures it)
        
        while remaining:
            # 1. Find a large clique in the remaining graph
            # Heuristic: Start with vertex of max degree, expand greedily
            
            # Calculate degrees in remaining graph
            deg = [0] * n
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            
            # Pick vertex with max degree
            if not any(deg):
                break # Should not happen if remaining is not empty
                
            start_v = max(range(n), key=lambda i: deg[i])
            
            clique = [start_v]
            # Candidates must be neighbors of all vertices in clique
            # and present in remaining graph
            candidates = set()
            for u, v in remaining:
                if u == start_v:
                    candidates.add(v)
                elif v == start_v:
                    candidates.add(u)
            
            while candidates:
                # Pick candidate with max degree within candidates
                # (i.e., number of neighbors in candidates)
                best_v = None
                best_deg = -1
                for v in candidates:
                    # Count neighbors in candidates
                    c_deg = 0
                    for u, w in remaining:
                        if u == v:
                            if w in candidates:
                                c_deg += 1
                        elif w == v:
                            if u in candidates:
                                c_deg += 1
                    if c_deg > best_deg:
                        best_deg = c_deg
                        best_v = v
                
                if best_v is None:
                    break
                    
                clique.append(best_v)
                candidates.remove(best_v)
                
                # Update candidates: must be connected to best_v
                # So remove any candidate not connected to best_v
                new_candidates = set()
                for v in candidates:
                    # Check if (best_v, v) is in remaining
                    if (best_v, v) in remaining or (v, best_v) in remaining:
                        new_candidates.add(v)
                candidates = new_candidates
            
            # Add clique to partition
            cliques.append(clique)
            
            # Remove all edges within clique from remaining
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    pair = (u, v) if u < v else (v, u)
                    remaining.discard(pair)
        
        return cliques

    adj1, edges1 = instance(32, 0.6, 601)
    adj2, edges2 = instance(64, 0.55, 602)
    
    cliques1 = solve(32, adj1, edges1)
    cliques2 = solve(64, adj2, edges2)
    
    return [cliques1, cliques2]
