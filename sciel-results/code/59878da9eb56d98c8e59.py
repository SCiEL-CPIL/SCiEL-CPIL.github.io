import time

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s limit
    
    n = 16
    # Construct K(8,8)
    # Partitions: 0-7 and 8-15
    edges = []
    for a in range(8):
        for b in range(8, 16):
            edges.append([a, b])
    
    # Verification is fast for 64 edges, so we run it to be safe against any logical oversight.
    if _verify(edges, n):
        return edges
    
    # Fallback: A star graph is a valid minimal solution (15 edges)
    # Diameter 2, all edges essential (removal disconnects leaves)
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
            q = [s]
            i = 0
            while i < len(q):
                u = q[i]
                i += 1
                for w in adj[u]:
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        q.append(w)
            if -1 in dist:
                return None  # Disconnected
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
