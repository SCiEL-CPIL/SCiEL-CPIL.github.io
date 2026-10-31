def _check_girth(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] * 2 >= g:
                # If dist[u] is large enough, any cycle through u will be >= g?
                # Cycle length L = dist[u] + dist[v] + 1.
                # If v is an ancestor, dist[v] < dist[u].
                # Min cycle length through u is roughly 2*dist[u] + 1? No.
                # If we find a back edge to v, L = dist[u] + dist[v] + 1.
                # Since dist[v] >= 0, L >= dist[u] + 1.
                # If dist[u] + 1 >= g, then L >= g.
                # So if dist[u] >= g - 1, we can stop.
                # But we also need to consider even cycles?
                # In a simple graph, cycles are simple.
                # Let's just check all edges from u.
                pass
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
