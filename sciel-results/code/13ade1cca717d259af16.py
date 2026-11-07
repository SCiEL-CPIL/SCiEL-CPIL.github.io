"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    # Initialize adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    added_edges = []
    
    def get_diameter_and_violations(adj):
        max_dist = 0
        violations = 0
        for i in range(n):
            # BFS
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for j in range(i+1, n):
                if dist[j] > max_dist:
                    max_dist = dist[j]
                if dist[j] > 4:
                    violations += 1
        return max_dist, violations

    # Deadline guard
    start_time = time.time()
    timeout = 50 # 50 seconds safety margin
    
    while time.time() - start_time < timeout:
        current_diam, current_viol = get_diameter_and_violations(adj)
        if current_diam <= 4:
            break
            
        best_score = float('inf')
        best_edge = None
        
        # Generate candidates
        # A = even, B = odd
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) in base_edges or (v, u) in base_edges:
                    continue
                # Check if already added
                if (u, v) in {tuple(e) for e in added_edges} or (v, u) in {tuple(e) for e in added_edges}:
                    continue
                # Bipartite check: u and v must have different parity
                if (u - v) % 2 != 1:
                    continue
                candidates.append((u, v))
                
        for u, v in candidates:
            # Temporarily add edge
            adj[u].add(v)
            adj[v].add(u)
            
            new_diam, new_viol = get_diameter_and_violations(adj)
            
            # Remove edge
            adj[u].remove(v)
            adj[v].remove(u)
            
            score = 10000 * new_diam + new_viol
            if score < best_score:
                best_score = score
                best_edge = (u, v)
                
        if best_edge is None:
            break
            
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added_edges.append([u, v])
        
    # Convert to list of lists of ints
    return [[int(u), int(v)] for u, v in added_edges]

# EVOLVE-BLOCK-END
