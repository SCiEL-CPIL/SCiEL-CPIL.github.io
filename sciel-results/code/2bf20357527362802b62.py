import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    def build_adj(added_edges):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_diameter(adj):
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_dist:
                            max_dist = dist[v]
                        if dist[v] > 4:
                            return max_dist
                        queue.append(v)
            if -1 in dist:
                return float('inf')
        return max_dist

    def is_valid(added_edges):
        adj = build_adj(added_edges)
        diam = get_diameter(adj)
        return diam <= 4

    # Initial solution: Hub at 12 connected to all odd vertices
    hub = 12
    initial_edges = set()
    for v in range(n):
        if v % 2 != 0 and v != hub:
            initial_edges.add((min(hub, v), max(hub, v)))
            
    # Prune edges
    added_edges = initial_edges
    improved = True
    while improved:
        improved = False
        for edge in list(added_edges):
            if not is_valid(added_edges - {edge}):
                continue
            added_edges = added_edges - {edge}
            improved = True
    
    # Ensure types are correct
    result = []
    for u, v in added_edges:
        result.append([int(u), int(v)])
        
    return result
