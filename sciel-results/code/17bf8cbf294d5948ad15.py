import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_diameter_exceeds(adj, n, threshold=4):
    """Returns True if diameter exceeds threshold, False otherwise. Early termination."""
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > threshold:
                        return True
                    q.append(v)
    return False

def _bfs_diameter_full(adj, n):
    """Returns (diameter, count_of_pairs_at_diameter)."""
    max_d = 0
    count_at_max = 0
    
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
            return 999, 0
            
        for i in range(n):
            if i > src:
                d = dist[i]
                if d > max_d:
                    max_d = d
                    count_at_max = 1
                elif d == max_d:
                    count_at_max += 1
    
    return max_d, count_at_max

def construct():
    n = P['n']
    base_edges = P['edges']
    base_set = set(tuple(e) for e in base_edges)
    
    # Pre-compute bipartite candidates
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_set or (v, u) in base_set:
                continue
            if u % 2 != v % 2:
                all_candidates.append((u, v))
    
    best_solution = None
    best_count = float('inf')
    
    deadline = time.time() + 80
    
    for attempt in range(300):
        if time.time() > deadline - 5:
            break
            
        current_added = []
        adj = {i: set() for i in range(n)}
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        curr_diam, curr_count = _bfs_diameter_full(adj, n)
        
        while curr_diam > 4:
            if time.time() > deadline - 5:
                break
                
            best_edge = None
            best_score = (curr_diam, curr_count)
            
            candidates = all_candidates[:]
            random.shuffle(candidates)
            
            for u, v in candidates:
                if v in adj[u]:
                    continue
                    
                adj[u].add(v)
                adj[v].add(u)
                
                if _bfs_diameter_exceeds(adj, n, threshold=4):
                    adj[u].remove(v)
                    adj[v].remove(u)
                    continue
                
                # Diameter is <= 4, get full score
                new_diam, new_count = _bfs_diameter_full(adj, n)
                new_score = (new_diam, new_count)
                
                adj[u].remove(v)
                adj[v].remove(u)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edge = (u, v)
                    
            if best_edge is None:
                break
                
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            current_added.append([u, v])
            curr_diam, curr_count = best_score
            
            if curr_diam <= 4:
                if len(current_added) < best_count:
                    best_count = len(current_added)
                    best_solution = [list(e) for e in current_added]
                break
    
    # Local search: try to remove edges
    if best_solution is not None and time.time() < deadline - 10:
        current = [list(e) for e in best_solution]
        adj = {i: set() for i in range(n)}
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in current:
            adj[u].add(v)
            adj[v].add(u)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(current) - 1, -1, -1):
                u, v = current[i]
                adj[u].remove(v)
                adj[v].remove(u)
                
                if not _bfs_diameter_exceeds(adj, n, threshold=4):
                    current.pop(i)
                    improved = True
                    break
                else:
                    adj[u].add(v)
                    adj[v].add(u)
        
        if len(current) < best_count:
            best_count = len(current)
            best_solution = current
    
    if best_solution is None:
        return []
        
    return [[int(u), int(v)] for u, v in best_solution]

# EVOLVE-BLOCK-END
