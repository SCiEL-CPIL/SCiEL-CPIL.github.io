import time

def construct():
    t0 = time.time()
    n = 16
    A = list(range(0, 8))
    B = list(range(8, 16))
    edges = [[a, b] for a in A for b in B]
    if _verify(edges, n):
        return edges
    # fallback: star
    return [[0, v] for v in range(1, n)]

def _verify(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def bfs_diameter(adj_list):
        maxd = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = [s]
            idx = 0
            while idx < len(q):
                u = q[idx]
                idx += 1
                du = dist[u]
                for w in adj_list[u]:
                    if dist[w] == -1:
                        dist[w] = du + 1
                        q.append(w)
            if dist[-1] == -1 or any(d == -1 for d in dist):
                return None
            maxd = max(maxd, max(dist))
        return maxd
    
    if bfs_diameter(adj) != 2:
        return False
    
    for (u, v) in edges:
        adj2 = [set(s) for s in adj]
        adj2[u].discard(v)
        adj2[v].discard(u)
        d2 = bfs_diameter(adj2)
        if d2 is not None and d2 <= 2:
            return False
    return True
