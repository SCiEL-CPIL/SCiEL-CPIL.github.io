"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        if u > v:
            u, v = v, u
        base_edges.add((u, v))
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Candidate edges: only different parity (preserves bipartiteness, hence triangle-free)
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) in base_edges:
                continue
            if (u % 2) != (v % 2):
                candidates.append((u, v))
                
    # Helper: Floyd-Warshall
    def compute_dist(adj_list):
        dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
            for v in adj_list[i]:
                dist[i][v] = 1
        for k in range(n):
            for i in range(n):
                if dist[i][k] == float('inf'):
                    continue
                for j in range(n):
                    if dist[k][j] == float('inf'):
                        continue
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def get_diameter(dist):
        diam = 0
        for i in range(n):
            for j in range(n):
                if dist[i][j] > diam:
                    diam = dist[i][j]
        return diam

    current_adj = [set(s) for s in adj]
    current_dist = compute_dist(current_adj)
    added = []
    added_set = set()
    deadline = time.time() + 50.0
    
    while time.time() < deadline:
        diam = get_diameter(current_dist)
        if diam <= 4:
            break
        
        best_edge = None
        best_diam = float('inf')
        best_dist = None
        
        for u, v in candidates:
            if (u, v) in added_set:
                continue
            
            temp_adj = [set(s) for s in current_adj]
            temp_adj[u].add(v)
            temp_adj[v].add(u)
            
            d = compute_dist(temp_adj)
            d_diam = get_diameter(d)
            
            if d_diam < best_diam:
                best_diam = d_diam
                best_edge = (u, v)
                best_dist = d
                
        if best_edge is None:
            break
            
        u, v = best_edge
        current_adj[u].add(v)
        current_adj[v].add(u)
        current_dist = best_dist
        added.append((u, v))
        added_set.add((u, v))
        
    result = [[u, v] for u, v in added]
    return result
# EVOLVE-BLOCK-END
