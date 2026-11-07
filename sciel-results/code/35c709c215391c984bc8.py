import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_adj = [set() for _ in range(n)]
    existing_edges = set()
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)
        existing_edges.add((min(u, v), max(u, v)))
    
    def bfs_diameter(adj):
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d > max_d:
                    max_d = d
        return max_d
    
    def is_triangle_free(adj):
        for u in range(n):
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        return False
        return True
    
    def check_solution(added_edges):
        adj = [set(s) for s in base_adj]
        seen = set()
        for u, v in added_edges:
            a, b = min(u, v), max(u, v)
            if (a, b) in seen:
                return False
            seen.add((a, b))
            adj[u].add(v)
            adj[v].add(u)
        
        if not is_triangle_free(adj):
            return False
        diam = bfs_diameter(adj)
        if diam > 4:
            return False
        return True
    
    adj = [set(s) for s in base_adj]
    added = []
    added_set = set()
    
    def is_valid_edge(u, v):
        if u == v:
            return False
        a, b = min(u, v), max(u, v)
        if (a, b) in existing_edges:
            return False
        if (a, b) in added_set:
            return False
        if adj[u] & adj[v]:
            return False
        return True
    
    deadline = time.time() + 50.0
    
    while time.time() < deadline:
        diam = bfs_diameter(adj)
        if diam <= 4:
            break
        
        best_edge = None
        best_new_diam = diam + 1
        
        for u in range(n):
            for v in range(u + 1, n):
                if not is_valid_edge(u, v):
                    continue
                adj[u].add(v)
                adj[v].add(u)
                new_diam = bfs_diameter(adj)
                adj[u].remove(v)
                adj[v].remove(u)
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added_set.add((min(u, v), max(u, v)))
        added.append([u, v])
    
    if added and check_solution(added):
        return [[int(u), int(v)] for u, v in added]
    
    return []
