import time
from collections import deque

def construct():
    deadline = time.time() + 80
    n = 24
    
    adj = [set() for _ in range(n)]
    base_set = set()
    for i in range(n - 1):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
        base_set.add((i, i + 1))
        base_set.add((i + 1, i))
    
    added = []
    
    def bfs_dist(src):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            node = q.popleft()
            for nb in adj[node]:
                if dist[nb] < 0:
                    dist[nb] = dist[node] + 1
                    q.append(nb)
        return dist
    
    def diameter():
        return max(max(bfs_dist(s)) for s in range(n))
    
    def far_count():
        c = 0
        for s in range(n):
            d = bfs_dist(s)
            c += sum(1 for x in d if x > 4)
        return c // 2
    
    def can_add(u, v):
        if (u, v) in base_set or (v, u) in base_set:
            return False
        return not (adj[u] & adj[v])
    
    # Greedy phase
    while time.time() < deadline:
        if diameter() <= 4:
            break
        cur = far_count()
        best = None
        best_g = -1
        for u in range(n):
            for v in range(u + 1, n):
                if not can_add(u, v):
                    continue
                adj[u].add(v)
                adj[v].add(u)
                g = cur - far_count()
                if g > best_g:
                    best_g = g
                    best = (u, v)
                adj[u].remove(v)
                adj[v].remove(u)
        if best is None:
            break
        u, v = best
        adj[u].add(v)
        adj[v].add(u)
        added.append([u, v])
    
    # Local search: try to remove edges
    for i in range(len(added) - 1, -1, -1):
        u, v = added[i]
        # Temporarily remove
        adj[u].remove(v)
        adj[v].remove(u)
        if diameter() <= 4:
            added.pop()
        else:
            adj[u].add(v)
            adj[v].add(u)
    
    return added
