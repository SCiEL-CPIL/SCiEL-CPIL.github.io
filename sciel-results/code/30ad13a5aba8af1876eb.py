def construct():
    import random
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
        return adj, edges

    def solve(n, adj, edges):
        # adj is a list of sets of neighbors
        # We will maintain a set of remaining edges
        remaining_edges = set(edges)
        # Also maintain a list of sets of remaining neighbors for fast intersection
        rem_adj = [set() for _ in range(n)]
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        cliques = []
        
        while remaining_edges:
            # Find the best edge: one with the most common neighbors in the remaining graph
            # Score = |N(u) ∩ N(v) ∩ Remaining|
            # Note: N(u) in remaining is rem_adj[u]
            
            best_edge = None
            max_common = -1
            
            # Iterate through remaining edges to find the best one
            # Optimization: We can sample or just iterate all. n=64, max edges ~2000.
            # Iterating 2000 edges is fast.
            for u, v in remaining_edges:
                # Common neighbors of u and v in remaining graph
                # Intersection of rem_adj[u] and rem_adj[v]
                common = len(rem_adj[u].intersection(rem_adj[v]))
                if common > max_common:
                    max_common = common
                    best_edge = (u, v)
            
            u, v = best_edge
            clique = [u, v]
            
            # Candidate set: vertices connected to all in clique via remaining edges
            # Start with common neighbors of u and v
            candidates = rem_adj[u].intersection(rem_adj[v])
            
            # Greedily add vertices from candidates
            while candidates:
                # Pick candidate w with max degree in the subgraph induced by candidates?
                # Or max common neighbors with current clique?
                # Since all candidates are connected to all clique members,
                # we just want to pick one that allows more future additions.
                # Heuristic: Pick w in candidates with max degree within candidates.
                
                best_w = None
                max_deg = -1
                for w in candidates:
                    # Degree of w in the subgraph induced by candidates
                    deg = len(rem_adj[w].intersection(candidates))
                    if deg > max_deg:
                        max_deg = deg
                        best_w = w
                
                if best_w is None:
                    break
                
                clique.append(best_w)
                
                # Update candidates: must be adjacent to best_w in remaining graph
                # candidates = candidates ∩ rem_adj[best_w]
                candidates = candidates.intersection(rem_adj[best_w])
            
            # Remove all edges within clique from remaining
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    edge = tuple(sorted((a, b)))
                    if edge in remaining_edges:
                        remaining_edges.remove(edge)
                        rem_adj[a].remove(b)
                        rem_adj[b].remove(a)
            
            cliques.append(clique)
        
        return cliques

    adj1, edges1 = instance(32, 0.6, 601)
    adj2, edges2 = instance(64, 0.55, 602)
    
    cliques1 = solve(32, adj1, edges1)
    cliques2 = solve(64, adj2, edges2)
    
    return [cliques1, cliques2]
