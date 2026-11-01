<<<<<<< SEARCH
def _compute_girth(adj, n):
    """Compute the girth (shortest cycle) of the graph."""
    min_girth = float('inf')
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            curr = queue.popleft()
            for nb in adj[curr]:
                if dist[nb] == -1:
                    dist[nb] = dist[curr] + 1
                    parent[nb] = curr
                    queue.append(nb)
                elif nb != parent[curr]:
                    # Found a cycle
                    cycle_len = dist[curr] + dist[nb] + 1
                    min_girth = min(min_girth, cycle_len)
    
    return min_girth if min_girth != float('inf') else n
=======
def _compute_girth(adj, n):
    """Compute the girth (shortest cycle) of the graph."""
    min_girth = float('inf')
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            curr = queue.popleft()
            for nb in adj[curr]:
                if dist[nb] == -1:
                    dist[nb] = dist[curr] + 1
                    parent[nb] = curr
                    queue.append(nb)
                elif nb != parent[curr]:
                    # Found a cycle
                    cycle_len = dist[curr] + dist[nb] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth <= 4: # Early exit for very small girth
                            return min_girth
                    if min_girth <= 6:
                        return min_girth
    
    return min_girth if min_girth != float('inf') else n

def _edges_to_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj
>>>>>>> REPLACE

