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
    for e in P['edges']:
        base_edges.add((min(e), max(e)))
        base_edges.add((max(e), min(e)))
    
    # Candidate edges: same-parity only
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if u % 2 == v % 2:
                if (u, v) not in base_edges and (v, u) not in base_edges:
                    candidates.append((u, v))
    
    def compute_diameter(added_edges):
        adj = [[] for _ in range(n)]
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in added_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        max_dist = max(max_dist, dist[nb])
                        q.append(nb)
        return max_dist
    
    # Greedy: start empty, add edges that most reduce diameter
    added = []
    while time.time() < deadline:
        d = compute_diameter(added)
        if d <= 4:
            break
        best_edge = None
        best_d = d
        for edge in candidates:
            if edge in added:
                continue
            td = compute_diameter(added + [edge])
            if td < best_d:
                best_d = td
                best_edge = edge
        if best_edge is None:
            break
        added.append(best_edge)
    
    # Minimize: try removing edges
    minimized = list(added)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(minimized)):
            if time.time() > deadline:
                break
            test = minimized[:i] + minimized[i+1:]
            if compute_diameter(test) <= 4:
                minimized = test
                changed = True
                break
    
    # Try to find even smaller solution via different greedy orderings
    # Try multiple random orderings
    import random
    best_solution = minimized
    
    for trial in range(50):
        if time.time() > deadline:
            break
        random.shuffle(candidates)
        added2 = []
        while time.time() < deadline:
            d = compute_diameter(added2)
            if d <= 4:
                break
            best_edge = None
            best_d = d
            for edge in candidates:
                if edge in added2:
                    continue
                td = compute_diameter(added2 + [edge])
                if td < best_d:
                    best_d = td
                    best_edge = edge
            if best_edge is None:
                break
            added2.append(best_edge)
        
        # Minimize
        min2 = list(added2)
        changed2 = True
        while changed2 and time.time() < deadline:
            changed2 = False
            for i in range(len(min2)):
                if time.time() > deadline:
                    break
                test = min2[:i] + min2[i+1:]
                if compute_diameter(test) <= 4:
                    min2 = test
                    changed2 = True
                    break
        
        if len(min2) < len(best_solution):
            best_solution = min2
    
    result = [[u, v] for u, v in best_solution]
    return result
# EVOLVE-BLOCK-END
