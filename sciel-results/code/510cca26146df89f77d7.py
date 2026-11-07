import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    
    adj = {i: set() for i in range(n)}
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def is_triangle_free(adj, u, v):
        if v in adj[u]:
            return False
        for w in adj[u]:
            if w in adj[v]:
                return False
        return True
    
    def bfs_diameter(adj, n):
        max_diam = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            if any(d == -1 for d in dist):
                return 999
            for d in dist:
                if d > max_diam:
                    max_diam = d
        return max_diam
    
    def is_valid(adj, n):
        if bfs_diameter(adj, n) > 4:
            return False
        for u in range(n):
            for v in adj[u]:
                for w in adj[u]:
                    if w != v and w in adj[v]:
                        return False
        return True
    
    deadline = time.time() + 85
    best = None
    best_cnt = float('inf')
    
    # Greedy: repeatedly add the edge that reduces diameter the most
    # while maintaining triangle-freeness
    import random
    
    for trial in range(300):
        if time.time() > deadline - 5:
            break
        
        curr = {i: set(adj[i]) for i in range(n)}
        added = []
        
        for step in range(10):
            if time.time() > deadline - 5:
                break
            
            d = bfs_diameter(curr, n)
            if d <= 4:
                break
            
            best_edge = None
            best_d = d
            
            # Try candidates in random order for diversity
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) in base_set or v in curr[u]:
                        continue
                    if not is_triangle_free(curr, u, v):
                        continue
                    
                    curr[u].add(v)
                    curr[v].add(u)
                    nd = bfs_diameter(curr, n)
                    curr[u].remove(v)
                    curr[v].remove(u)
                    
                    if nd < best_d:
                        best_d = nd
                        best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            curr[u].add(v)
            curr[v].add(u)
            added.append([u, v])
        
        if is_valid(curr, n) and len(added) < best_cnt:
            best_cnt = len(added)
            best = [list(e) for e in added]
            if best_cnt <= 5:
                break
    
    if best is None:
        return []
    return [[int(u), int(v)] for u, v in best]

# EVOLVE-BLOCK-END
