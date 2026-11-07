import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    t0 = time.time()
    deadline = 50.0  # Safety margin
    n = P['n']
    base_edges = P['edges']
    
    # Base adjacency
    base_adj = [set() for _ in range(n)]
    base_edge_set = set()
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
        base_edge_set.add((min(u, v), max(u, v)))
    
    # Precompute candidates that are triangle-safe w.r.t base
    # This filters out many edges that can never be added
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edge_set:
                continue
            # Check triangle safety w.r.t base
            # If there is w in base_adj[u] and base_adj[v], adding (u,v) creates triangle u-w-v
            # Note: w cannot be u or v obviously.
            if base_adj[u] & base_adj[v]:
                continue
            candidates.append((u, v))
            
    def get_dist_matrix(adj):
        dist_matrix = [[-1] * n for _ in range(n)]
        for s in range(n):
            dist_matrix[s][s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist_matrix[s][v] == -1:
                        dist_matrix[s][v] = dist_matrix[s][u] + 1
                        q.append(v)
        return dist_matrix

    def get_diameter(dist_matrix):
        diam = 0
        worst_pair = (0, 0)
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_matrix[i][j]
                if d > diam:
                    diam = d
                    worst_pair = (i, j)
        return diam, worst_pair

    def is_safe_to_add(u, v, adj, edge_set):
        if (min(u, v), max(u, v)) in edge_set:
            return False
        # Check triangle in current graph
        if adj[u] & adj[v]:
            return False
        return True

    best_result = None
    best_count = float('inf')
    
    # Run greedy multiple times with slight variations or just once if deterministic is fast enough
    # For n=24, one good greedy run should be sufficient and fast.
    
    for attempt in range(10): # Try a few times, maybe randomize candidate order? No, keep deterministic first.
        if time.time() - t0 > deadline:
            break
            
        added = []
        adj = [set(base_adj[i]) for i in range(n)]
        edge_set = set(base_edge_set)
        
        # Greedy Loop
        while True:
            dist_matrix = get_dist_matrix(adj)
            diam, (wu, wv) = get_diameter(dist_matrix)
            
            if diam <= 4:
                break
            
            # Find best edge to add
            best_edge = None
            best_metric = None
            
            # To speed up, we can prioritize edges that are "far" from current worst pair?
            # Or just iterate all candidates.
            # Number of candidates is roughly n^2/2 - n = ~276.
            # 276 * (BFS cost) is acceptable.
            
            for (u, v) in candidates:
                if not is_safe_to_add(u, v, adj, edge_set):
                    continue
                
                # Tentative add
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((min(u, v), max(u, v)))
                
                new_dist_matrix = get_dist_matrix(adj)
                new_diam, (nwu, nwv) = get_diameter(new_dist_matrix)
                
                # Metric: minimize new diameter, then minimize max distance of previous worst pair?
                # Or just minimize new diameter.
                # If tie, prefer edge that reduces distance of (wu, wv) most?
                d_wu_wv = new_dist_matrix[wu][wv]
                
                metric = (new_diam, d_wu_wv)
                
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_edge = (u, v)
                
                # Tentative remove
                adj[u].remove(v)
                adj[v].remove(u)
                edge_set.discard((min(u, v), max(u, v)))
            
            if best_edge is None:
                # No edge can be added? Should not happen if graph is connected and diam > 4
                break
            
            u, v = best_edge
            added.append((u, v))
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))
            
        # Local Search: Remove edges
        improved = True
        while improved and time.time() - t0 < deadline:
            improved = False
            for i in range(len(added)):
                u, v = added[i]
                # Try removing
                adj[u].remove(v)
                adj[v].remove(u)
                edge_set.discard((min(u, v), max(u, v)))
                
                dist_matrix = get_dist_matrix(adj)
                diam, _ = get_diameter(dist_matrix)
                
                if diam <= 4:
                    # Keep removed
                    added.pop(i)
                    improved = True
                    break # Restart loop as indices changed
                else:
                    # Re-add
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add((min(u, v), max(u, v)))
        
        if len(added) < best_count:
            best_count = len(added)
            best_result = [list(e) for e in added]
            
        # If we found a solution with 0 edges (impossible for path 24) or 1, 2, 3...
        # We can stop if we find a very small number, e.g., 3 or 4.
        if best_count <= 3:
            break

    if best_result is None:
        return []
        
    # Ensure types
    return [[int(u), int(v)] for u, v in best_result]
