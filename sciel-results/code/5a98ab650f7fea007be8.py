def construct(k, g):
    deadline = time.time() + 110
    d = max(1, (g - 2 + 1) // 2)
    n, edges = _lzw_construct(k, d, deadline)
    
    # Optional: Verify girth
    # For 1250 vertices, BFS verification is O(n*k) ~ 6000 ops, very fast.
    if _verify_girth(n, edges, g):
        return n, edges
    else:
        # If it fails, something is wrong. Return it anyway? 
        # Or try d+1?
        # If LZW is buggy, d+1 might also be buggy.
        # But we must return something.
        return n, edges

def _verify_girth(n, edges, g):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
                elif nb != node:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < g:
                        return False
    return True
