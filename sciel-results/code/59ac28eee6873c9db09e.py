def construct():
    import random
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        # Build adjacency matrix
        adj = [[False] * n for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = adj[v][u] = True
                    edges.append((u, v))
        return adj, edges
    
    def solve(n, adj, edges):
        covered = set()
        for e in edges:
            covered.add(tuple(sorted(e)))
        remaining = set(covered)
        cliques = []
        
        # Precompute for speed: for each pair, whether it's an edge
        # We'll work with the remaining set
        
        while remaining:
            # Pick an arbitrary remaining edge
            edge = next(iter(remaining))
            u, v = edge
            # Build clique starting with {u, v}
            clique = [u, v]
            # Try to add vertices in order
            for w in range(n):
                if w == u or w == v:
                    continue
                # Check if w can be added: for all x in clique, (w,x) must be in remaining
                can_add = True
                for x in clique:
                    pair = tuple(sorted((w, x)))
                    if pair not in remaining:
                        can_add = False
                        break
                if can_add:
                    clique.append(w)
            
            # Remove all edges within clique from remaining
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    pair = tuple(sorted((clique[i], clique[j])))
                    remaining.discard(pair)
            
            cliques.append(clique)
        
        return cliques
    
    adj1, edges1 = instance(32, 0.6, 601)
    adj2, edges2 = instance(64, 0.55, 602)
    
    cliques1 = solve(32, adj1, edges1)
    cliques2 = solve(64, adj2, edges2)
    
    return [cliques1, cliques2]
