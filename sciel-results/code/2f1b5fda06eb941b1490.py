def _girth_at_least(n, edges, g):
    if g <= 2:
        return True
        
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    max_depth = g // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] >= max_depth:
                continue
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                else:
                    # v is visited
                    # Skip if v is parent of u
                    if v == parent[u]:
                        continue
                    # Skip if u is parent of v
                    if parent[v] == u:
                        continue
                        
                    # Otherwise, it's a cross edge or back edge to non-parent
                    # Cycle length is <= dist[u] + dist[v] + 1
                    if dist[u] + dist[v] + 1 < g:
                        return False
                            
    return True
