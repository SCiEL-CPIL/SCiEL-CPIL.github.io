import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = P['n']
    base_edges = P['edges']
    
    existing = set()
    for u, v in base_edges:
        existing.add((min(u, v), max(u, v)))
    
    def validate(added):
        edge_set = set()
        for u, v in added:
            mn, mx = min(u, v), max(u, v)
            if (mn, mx) in existing or (mn, mx) in edge_set:
                return False
            edge_set.add((mn, mx))
        
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added:
            adj[u].add(v)
            adj[v].add(u)
        
        for u in range(n):
            for v in adj[u]:
                if v > u and (adj[u] & adj[v]):
                    return False
        
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        if not all(visited):
            return False
        
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if any(d > 4 for d in dist):
                return False
        return True
    
    # Try 6-edge solutions with systematic search
    # Key insight: we need to cover the "long" pairs
    # Pairs with path distance > 4 need shortcuts
    
    # Let me try specific 6-edge configurations
    test_sols = [
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 11), (17, 10)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 11), (17, 12)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 11), (17, 13)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 11), (17, 14)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 11), (17, 15)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 11), (17, 16)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 10), (17, 11)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 10), (17, 12)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 10), (17, 13)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 12), (17, 13)],
        [(6, 0), (6, 12), (6, 17), (17, 23), (17, 13), (17, 14)],
        [(5, 0), (5, 11), (5, 17), (17, 23), (17, 11), (17, 12)],
        [(5, 0), (5, 11), (5, 17), (17, 23), (17, 12), (17, 13)],
        [(7, 0), (7, 13), (7, 17), (17, 23), (17, 11), (17, 10)],
        [(7, 0), (7, 13), (7, 17), (17, 23), (17, 11), (17, 12)],
        [(6, 0), (6, 11), (6, 17), (17, 23), (17, 12), (17, 10)],
        [(6, 0), (6, 11), (6, 17), (17, 23), (17, 12), (17, 11)],
    ]
    
    for sol in test_sols:
        if time.time() > deadline:
            break
        if validate(sol):
            return [[int(u), int(v)] for u, v in sol]
    
    # If no 6-edge found, return 7-edge fallback
    fallback = [[11, 1], [11, 4], [11, 7], [11, 15], [11, 18], [11, 21], [11, 23]]
    return [[int(u), int(v)] for u, v in fallback]
