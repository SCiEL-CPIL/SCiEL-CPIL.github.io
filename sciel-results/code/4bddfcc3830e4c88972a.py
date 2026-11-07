# EVOLVE-BLOCK-START
from itertools import combinations
from collections import deque
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in base_edges:
                candidates.append((u, v))
                
    candidates.sort(key=lambda x: x[1]-x[0], reverse=True)
    
    def is_triangle_free(adj, u, v):
        return len(adj[u] & adj[v]) == 0
        
    def check_diameter(adj, max_diam=4):
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > max_diam:
                            return False
                        q.append(nb)
            if any(d == -1 for d in dist):
                return False
        return True
        
    best_solution = None
    start_time = time.time()
    
    # Try k=5 with a time limit
    for k in [5, 6]:
        nodes_explored = 0
        limit = 100000 if k == 5 else 500000
        timeout = 10 if k == 5 else 30
        
        def backtrack(selected, idx, adj_copy):
            nonlocal found, nodes_explored
            if found is not None:
                return
            if time.time() - start_time > timeout:
                return
            nodes_explored += 1
            if nodes_explored > limit:
                return
            if len(selected) == k:
                if check_diameter(adj_copy):
                    found = list(selected)
                return
            if idx >= len(candidates):
                return
                
            # Skip
            backtrack(selected, idx+1, adj_copy)
            if found is not None: return
            
            # Add
            u, v = candidates[idx]
            if is_triangle_free(adj_copy, u, v):
                adj_copy[u].add(v)
                adj_copy[v].add(u)
                selected.append((u, v))
                
                backtrack(selected, idx+1, adj_copy)
                
                selected.pop()
                adj_copy[u].remove(v)
                adj_copy[v].remove(u)
                
        found = None
        adj_copy = [set(s) for s in adj]
        backtrack([], 0, adj_copy)
        
        if found:
            best_solution = found
            break
        elif time.time() - start_time > 40:
            break
            
    if best_solution:
        return [[u, v] for u, v in best_solution]
        
    # Fallback to greedy
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    added_edges = []
    remaining = set(candidates)
    
    def compute_diameter(adj):
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        max_d = max(max_d, dist[nb])
                        q.append(nb)
        return max_d
        
    while True:
        diam = compute_diameter(adj)
        if diam <= 4:
            break
        if not remaining:
            break
        best_edge = None
        best_new_diam = float('inf')
        for (u, v) in remaining:
            if is_triangle_free(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                new_diam = compute_diameter(adj)
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
        added_edges.append((u, v))
        remaining.discard((u, v))
        
    return [[u, v] for u, v in added_edges]

# EVOLVE-BLOCK-END
