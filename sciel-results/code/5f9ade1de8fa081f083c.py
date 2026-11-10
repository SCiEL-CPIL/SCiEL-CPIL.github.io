import time
from collections import deque

def construct():
    t0 = time.time()
    deadline = 85.0
    n = 16
    
    # Theoretical Maximum for Triangle-Free Graphs (Mantel's Theorem) is n^2/4.
    # For n=16, max edges = 64.
    # Any graph with a triangle has edges that are not essential (common neighbor provides path of length 2).
    # Thus, the graph must be triangle-free.
    # K(8,8) is the unique triangle-free graph with 64 edges.
    # In K(8,8), removing any edge (u,v) increases d(u,v) to 3, so diameter becomes 3 > 2.
    # Thus, all edges are essential.
    
    A = list(range(0, 8))
    B = list(range(8, 16))
    edges = [[a, b] for a in A for b in B]
    
    # Optional: Verify to be safe, though mathematical proof holds.
    # Verification is O(E * (V+E)) which is fast for 64 edges.
    if _verify(edges, n):
        return edges
    
    # Fallback if verification fails (should not happen)
    return [[0, v] for v in range(1, n)]

def _verify(edges, n):
    adj = {v: set() for v in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def get_diameter():
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
                return None # Disconnected
            maxd = max(maxd, max(dist))
        return maxd
    
    if get_diameter() != 2:
        return False
        
    for u, v in edges:
        # Temporarily remove edge
        adj[u].remove(v)
        adj[v].remove(u)
        
        d2 = get_diameter()
        
        # Restore edge
        adj[u].add(v)
        adj[v].add(u)
        
        if d2 is not None and d2 <= 2:
            return False
            
    return True
