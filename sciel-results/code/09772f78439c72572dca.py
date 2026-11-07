import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))
    
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if (u, v) in base_edges and u != v:
                adj[u].add(v)
    
    def is_triangle_free(u, v):
        return len(adj[u] & adj[v]) == 0
    
    def compute_diameter():
        max_dist = 0
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
            m = max(dist)
            if m > max_dist:
                max_dist = m
        return max_dist
    
    # Generate all candidate edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges and (v, u) not in base_edges:
                candidates.append((u, v))
    
    added = []
    current_diam = compute_diameter()
    
    # Greedy Phase
    while current_diam > 4:
        if time.time() > deadline:
            break
        
        best_edge = None
        best_diam = current_diam
        best_sum_dist = float('inf')
        
        for edge in candidates:
            if time.time() > deadline:
                break
            u, v = edge
            if not is_triangle_free(u, v):
                continue
            
            adj[u].add(v)
            adj[v].add(u)
            d = compute_diameter()
            
            # Secondary metric: sum of all pairwise distances (to break ties and improve robustness)
            sum_dist = 0
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                q = deque([src])
                while q:
                    u2 = q.popleft()
                    for v2 in adj[u2]:
                        if dist[v2] == -1:
                            dist[v2] = dist[u2] + 1
                            q.append(v2)
                sum_dist += sum(dist)
            
            adj[u].remove(v)
            adj[v].remove(u)
            
            if d < best_diam or (d == best_diam and sum_dist < best_sum_dist):
                best_diam = d
                best_sum_dist = sum_dist
                best_edge = edge
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added.append([u, v])
        candidates.remove(best_edge)
        current_diam = best_diam
        
    # If still > 4, try Swap Optimization
    if current_diam > 4 and time.time() < deadline:
        # Try to swap one added edge for two others? Or just find a better set.
        # Simple local search: remove an edge, try to add a different one.
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(added)):
                if time.time() > deadline:
                    break
                u1, v1 = added[i]
                # Remove edge i
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                added.remove(added[i])
                candidates.append((u1, v1))
                
                diam_removed = compute_diameter()
                
                # Try to add a new edge to improve
                best_new_edge = None
                best_new_diam = diam_removed
                for edge in candidates:
                    if time.time() > deadline:
                        break
                    u2, v2 = edge
                    if not is_triangle_free(u2, v2):
                        continue
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    d = compute_diameter()
                    if d < best_new_diam:
                        best_new_diam = d
                        best_new_edge = edge
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    
                if best_new_edge is not None:
                    u2, v2 = best_new_edge
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    added.append([u2, v2])
                    candidates.remove(best_new_edge)
                    improved = True
                else:
                    # Restore original
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    added.append([u1, v1])
                    candidates.remove((u1, v1))
                    
    # Final check
    # Ensure types are correct
    result = []
    for edge in added:
        if isinstance(edge, tuple):
            result.append([int(edge[0]), int(edge[1])])
        else:
            result.append([int(edge[0]), int(edge[1])])
    return result
