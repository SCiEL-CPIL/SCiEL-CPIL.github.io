import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        if u > v: u, v = v, u
        base_edges.add((u, v))
    
    # Generate valid candidates: opposite parity, not in base
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u + v) % 2 == 1 and (u, v) not in base_edges:
                candidates.append((u, v))
    
    def get_diameter_and_pair(edges_set):
        adj = {i: [] for i in range(n)}
        for u, v in edges_set:
            adj[u].append(v)
            adj[v].append(u)
        
        max_dist = 0
        max_pair = (0, 0)
        
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
                            max_pair = (src, nb)
                        q.append(nb)
        
        return max_dist, max_pair

    current_edges = set(base_edges)
    added = []
    
    # Greedy loop
    # Safety limit to prevent infinite loops if something goes wrong
    max_iterations = 50 
    for _ in range(max_iterations):
        diam, (s, t) = get_diameter_and_pair(current_edges)
        if diam <= 4:
            break
        
        # Find best candidate to add
        best_cand = None
        best_new_dist = float('inf')
        
        # We want to reduce distance between s and t
        # Distance via candidate (u, v) is min(d(s,u)+1+d(v,t), d(s,v)+1+d(u,t))
        # We need distances from s and t to all nodes in current graph
        adj = {i: [] for i in range(n)}
        for u, v in current_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        def bfs_dist(start):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        q.append(nb)
            return dist

        dist_s = bfs_dist(s)
        dist_t = bfs_dist(t)
        
        for u, v in candidates:
            if (u, v) in current_edges:
                continue
            # Check if adding (u,v) reduces distance between s and t
            # Path s->...->u -> v ->...->t or s->...->v -> u ->...->t
            d1 = dist_s[u] + 1 + dist_t[v]
            d2 = dist_s[v] + 1 + dist_t[u]
            cand_dist = min(d1, d2)
            
            if cand_dist < best_new_dist:
                best_new_dist = cand_dist
                best_cand = (u, v)
        
        if best_cand is None:
            # Should not happen if graph is connected and candidates exist
            break
            
        # Add the best candidate
        current_edges.add(best_cand)
        added.append(list(best_cand))
        
    # Final check: if we didn't reach diameter 4, try a fallback
    # The fallback is to just add the longest edges until diameter <= 4
    if get_diameter_and_pair(current_edges)[0] > 4:
        added = []
        current_edges = set(base_edges)
        # Sort candidates by length descending
        sorted_cands = sorted(candidates, key=lambda x: x[1] - x[0], reverse=True)
        for u, v in sorted_cands:
            current_edges.add((u, v))
            added.append([u, v])
            if get_diameter_and_pair(current_edges)[0] <= 4:
                break
    
    return added
