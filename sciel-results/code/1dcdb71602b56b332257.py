import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        if u > v: u, v = v, u
        base_edges.add((u, v))
    
    # Generate all valid candidate edges: even-odd, not in base
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u + v) % 2 == 1 and (u, v) not in base_edges:
                candidates.append((u, v))
    
    def compute_diameter(edges_set):
        adj = [[] for _ in range(n)]
        for u, v in edges_set:
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
                        if dist[nb] > max_dist:
                            max_dist = dist[nb]
                            if max_dist > 4:
                                return max_dist  # early exit
                        q.append(nb)
        return max_dist
    
    current_edges = set(base_edges)
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline
    
    # Try k = 1, 2, 3, ...
    for k in range(1, 8):
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
            new_edges = current_edges | set(combo)
            diam = compute_diameter(new_edges)
            if diam <= 4:
                return [list(e) for e in combo]
        if time.time() > deadline:
            break
    
    # Fallback: greedy approach
    added = []
    current_edges = set(base_edges)
    for _ in range(20):
        if time.time() > deadline:
            break
        diam = compute_diameter(current_edges)
        if diam <= 4:
            break
        best_cand = None
        best_diam = float('inf')
        for u, v in candidates:
            if (u, v) in current_edges:
                continue
            new_edges = current_edges | {(u, v)}
            d = compute_diameter(new_edges)
            if d < best_diam:
                best_diam = d
                best_cand = (u, v)
            if time.time() > deadline:
                break
        if best_cand is None:
            break
        current_edges.add(best_cand)
        added.append(list(best_cand))
    
    return added
