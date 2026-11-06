def construct():
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    best_count = 0
    best_edges = []
    path_edges = []
    
    def is_3_colorable(adj):
        colors = [-1] * n
        vertices = list(range(n))
        vertices.sort(key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = vertices[idx]
            used = 0
            # Optimization: only check neighbors
            neighbors_mask = adj[v]
            # Iterate bits? Or just loop 0..15
            for u in range(n):
                if neighbors_mask & (1 << u):
                    if colors[u] != -1:
                        used |= (1 << colors[u])
            
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    def search(idx, adj, count):
        nonlocal best_count, best_edges
        
        if count + (num_edges - idx) <= best_count:
            return
            
        # Prune if max supergraph is 3-colorable
        temp_adj = adj[:]
        for i in range(idx, num_edges):
            u, v = edges[i]
            if not (temp_adj[u] & temp_adj[v]):
                temp_adj[u] |= (1 << v)
                temp_adj[v] |= (1 << u)
                
        if is_3_colorable(temp_adj):
            return

        if idx == num_edges:
            if count > best_count:
                best_count = count
                best_edges = path_edges.copy()
            return

        u, v = edges[idx]
        
        # Include
        if not (adj[u] & adj[v]):
            path_edges.append((u, v))
            new_adj = adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            search(idx + 1, new_adj, count + 1)
            path_edges.pop()
            
        # Exclude
        search(idx + 1, adj, count)

    initial_adj = [0] * n
    search(0, initial_adj, 0)
    
    return [list(e) for e in best_edges]
