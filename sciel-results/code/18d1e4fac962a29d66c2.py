import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 85
    
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if v - u >= 3:
                candidates.append((u, v))
    candidates.sort(key=lambda e: -(e[1] - e[0]))
    
    def bfs_max_dist(adj):
        max_d = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_d:
                            max_d = dist[v]
                        if dist[v] > 4:
                            return dist[v]
                        queue.append(v)
        return max_d
    
    # Greedy: add edges that most reduce max distance
    greedy_adj = [set(s) for s in base_adj]
    greedy_added = []
    while bfs_max_dist(greedy_adj) > 4 and len(greedy_added) < 15:
        best_edge = None
        best_diam = float('inf')
        for u, v in candidates:
            if u in greedy_adj[v]:
                continue
            if greedy_adj[u] & greedy_adj[v]:
                continue
            greedy_adj[u].add(v)
            greedy_adj[v].add(u)
            d = bfs_max_dist(greedy_adj)
            if d < best_diam:
                best_diam = d
                best_edge = (u, v)
            greedy_adj[u].remove(v)
            greedy_adj[v].remove(u)
        if best_edge is None:
            break
        u, v = best_edge
        greedy_adj[u].add(v)
        greedy_adj[v].add(u)
        greedy_added.append((u, v))
    
    best = greedy_added[:]
    best_count = len(greedy_added)
    
    # Backtrack to improve
    def backtrack(idx, adj, added):
        nonlocal best, best_count
        if time.time() > deadline:
            return
        
        diam = bfs_max_dist(adj)
        if diam <= 4:
            if len(added) < best_count:
                best = added[:]
                best_count = len(added)
            return
        
        if len(added) >= best_count:
            return
        
        for i in range(idx, len(candidates)):
            if time.time() > deadline:
                return
            u, v = candidates[i]
            if u in adj[v]:
                continue
            if adj[u] & adj[v]:
                continue
            
            adj[u].add(v)
            adj[v].add(u)
            added.append((u, v))
            
            backtrack(i + 1, adj, added)
            
            adj[u].remove(v)
            adj[v].remove(u)
            added.pop()
    
    backtrack(0, [set(s) for s in base_adj], [])
    
    if best is None:
        return []
    
    return [[int(u), int(v)] for u, v in best]
