    def local_search(edges, adj, max_steps=1000):
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if valid_add(u, v, adj):
                            edges.append((u, v))
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True
        
        # Random walk / Swap phase
        for _ in range(max_steps):
            if not edges: break
            # Remove random edge
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            edges.pop(idx)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            # Try to add any edge
            added = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        if valid_add(i, j, adj):
                            edges.append((i, j))
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            added = True
                            break
                if added: break
            
            # If we added an edge, we are at same count.
            # If we didn't, we lost an edge. 
            # If we lost an edge, we should probably revert if we don't gain back soon?
            # For simplicity, if we didn't add back, revert.
            if not added:
                edges.append((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        return edges, adj
