def find_long_cycle(edges):
    if not edges:
        return None
    
    adj = {}
    for u, v in edges:
        if u not in adj: adj[u] = set()
        if v not in adj: adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)
    
    if not adj:
        return None
    
    start_node = max(adj.keys(), key=lambda x: len(adj[x]))
    num_nodes = len(adj)
    
    best_cycle = None
    best_len = 0
    
    for neighbor in adj[start_node]:
        edge_to_remove = tuple(sorted((start_node, neighbor)))
        if edge_to_remove not in edges:
            continue
            
        temp_edges = edges - {edge_to_remove}
        
        current_path = [neighbor]
        visited = {neighbor}
        
        def dfs(node, path, vis):
            nonlocal best_cycle, best_len
            
            # Pruning
            if len(path) + (num_nodes - len(vis)) <= best_len:
                return
            
            if node == start_node and len(path) > 1:
                if len(path) > best_len:
                    best_len = len(path)
                    best_cycle = path[:]
                return
            
            for nxt in adj[node]:
                if nxt in vis:
                    continue
                e1 = (node, nxt)
                e2 = (nxt, node)
                if e1 not in temp_edges and e2 not in temp_edges:
                    continue
                
                vis.add(nxt)
                path.append(nxt)
                dfs(nxt, path, vis)
                path.pop()
                vis.remove(nxt)
                
        dfs(neighbor, current_path, visited)
        
    return best_cycle
