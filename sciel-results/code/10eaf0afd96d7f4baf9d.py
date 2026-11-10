import time
from collections import deque

def construct():
    t0 = time.time()
    deadline = 85.0
    n = 16
    
    # Construct K_8,8
    A = list(range(0, 8))
    B = list(range(8, 16))
    edges = []
    for a in A:
        for b in B:
            edges.append([int(a), int(b)])
            
    # Verify
    if _verify(edges, n):
        return edges

    # Fallback: K_2,14 (valid, 28 edges)
    edges_fallback = []
    A2 = [0, 1]
    B2 = list(range(2, 16))
    for a in A2:
        for b in B2:
            edges_fallback.append([int(a), int(b)])
            
    if _verify(edges_fallback, n):
        return edges_fallback

    # Final Fallback: Star (15 edges)
    edges_star = [[int(0), int(v)] for v in range(1, n)]
    return edges_star

def _bfs_diameter(adj, n):
    maxd = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    q.append(w)
        if -1 in dist:
            return None
        maxd = max(maxd, max(dist))
    return maxd

def _verify(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check diameter == 2
    d = _bfs_diameter(adj, n)
    if d != 2:
        return False
    
    # Check criticality
    for u, v in edges:
        adj[u].remove(v)
        adj[v].remove(u)
        
        d2 = _bfs_diameter(adj, n)
        
        # Restore
        adj[u].add(v)
        adj[v].add(u)
        
        # If connected and diameter <= 2, edge is NOT critical
        if d2 is not None and d2 <= 2:
            return False
            
    return True
