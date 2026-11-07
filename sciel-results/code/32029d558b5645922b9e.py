import sys
import time
from collections import deque

def construct():
    P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}
    n = P['n']
    start_time = time.time()
    deadline = 85.0
    
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    # Pre-compute all valid cross-parity edges (not in base)
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 != 0:
                if (u, v) not in base_edges:
                    all_candidates.append((u, v))
                    
    def get_diameter(adj):
        max_dist = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            if -1 in dist:
                return float('inf')
            max_dist = max(max_dist, max(dist))
        return max_dist

    initial_adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        initial_adj[u].add(v)
        initial_adj[v].add(u)
        
    added_edges = set()
    
    # Greedy Phase
    while True:
        if time.time() > deadline - 5.0:
            break
            
        adj = [set(s) for s in initial_adj]
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        diam = get_diameter(adj)
        if diam <= 4:
            break
            
        best_edge = None
        best_diam = float('inf')
        
        for u, v in all_candidates:
            if (u, v) in added_edges or (v, u) in added_edges:
                continue
                
            # Since we only add cross-parity edges, triangle-free is guaranteed.
            # We just need to check if it improves diameter.
            adj[u].add(v)
            adj[v].add(u)
            d = get_diameter(adj)
            adj[u].remove(v)
            adj[v].remove(u)
            
            if d < best_diam:
                best_diam = d
                best_edge = (u, v)
                
        if best_edge is None:
            break
            
        added_edges.add(best_edge)
        
    # Pruning Phase
    changed = True
    while changed:
        if time.time() > deadline - 5.0:
            break
        changed = False
        for e in list(added_edges):
            added_edges.remove(e)
            adj = [set(s) for s in initial_adj]
            for u, v in added_edges:
                adj[u].add(v)
                adj[v].add(u)
            d = get_diameter(adj)
            if d > 4:
                added_edges.add(e)
            else:
                changed = True
                break
                
    result = [[u, v] for u, v in added_edges]
    return result
