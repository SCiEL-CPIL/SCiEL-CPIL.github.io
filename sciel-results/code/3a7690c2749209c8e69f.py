"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u,v), max(u,v)))
    
    candidates = []
    for u in range(n):
        for v in range(u+3, n):
            if (u, v) not in base_edges:
                candidates.append((u, v))
    
    def build_adj(all_edges):
        adj = [set() for _ in range(n)]
        for u, v in all_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free(adj):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[u]:
                        if w > v and w in adj[v]:
                            return False
        return True
    
    def get_diameter(adj):
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
                return -1
            max_diam = max(max_diam, max(dist))
        return max_diam
    
    def is_valid(added):
        all_edges = list(P['edges']) + list(added)
        edge_set = set()
        for u, v in all_edges:
            key = (min(u,v), max(u,v))
            if key in edge_set:
                return False
            edge_set.add(key)
        adj = build_adj(all_edges)
        if not is_triangle_free(adj):
            return False
        diam = get_diameter(adj)
        return 0 <= diam <= 4
    
    # Greedy construction
    current = list(P['edges'])
    added = []
    adj = build_adj(current)
    diam = get_diameter(adj)
    
    while diam > 4 and time.time() < deadline:
        best_edge = None
        best_diam = diam
        for (u, v) in candidates:
            if (u, v) in base_edges:
                continue
            new_adj = build_adj(current + [(u, v)])
            if not is_triangle_free(new_adj):
                continue
            d = get_diameter(new_adj)
            if d < best_diam:
                best_diam = d
                best_edge = (u, v)
        if best_edge is None:
            break
        current.append(best_edge)
        added.append(best_edge)
        adj = build_adj(current)
        diam = best_diam
    
    best_solution = list(added)
    
    # Try to remove edges
    if diam <= 4 and time.time() < deadline:
        while time.time() < deadline:
            improved = False
            for i in range(len(best_solution)):
                test = best_solution[:i] + best_solution[i+1:]
                if is_valid(test):
                    best_solution = test
                    improved = True
                    break
            if not improved:
                break
    
    # Fallback
    if not best_solution or not is_valid(best_solution):
        best_solution = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 23), (0, 23)]
    
    result = []
    for u, v in best_solution:
        result.append([int(u), int(v)])
    return result

# EVOLVE-BLOCK-END

