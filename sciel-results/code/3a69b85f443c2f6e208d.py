import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit
    
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
                
    def bfs_dists(src, adj):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            node = q.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    q.append(nb)
        return dist

    def compute_diameter_and_pair(edges_set):
        adj = {i: [] for i in range(n)}
        for u, v in edges_set:
            adj[u].append(v)
            adj[v].append(u)
            
        max_dist = 0
        max_pair = (0, 0)
        for src in range(n):
            dist = bfs_dists(src, adj)
            for d in dist:
                if d > max_dist:
                    max_dist = d
                    max_pair = (src, dist.index(d))
        return max_dist, max_pair

    current_edges = set(base_edges)
    added = []
    
    # Greedy loop
    max_iterations = 50 
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
            
        diam, (s, t) = compute_diameter_and_pair(current_edges)
        if diam <= 4:
            break
        
        # Get distances from s and t
        adj = {i: [] for i in range(n)}
        for u, v in current_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        dist_s = bfs_dists(s, adj)
        dist_t = bfs_dists(t, adj)
        
        # Score candidates
        scored_candidates = []
        for u, v in candidates:
            if (u, v) in current_edges:
                continue
            
            # Estimate new distance between s and t
            # Path s->u->v->t or s->v->u->t
            d1 = dist_s[u] + 1 + dist_t[v]
            d2 = dist_s[v] + 1 + dist_t[u]
            new_d_st = min(diam, d1, d2)
            
            # Also consider if u or v are close to s or t
            # Heuristic: prefer edges that are "central"
            # Or just use new_d_st as primary key
            scored_candidates.append((new_d_st, u, v))
        
        # Sort by estimated new diameter (lower is better)
        scored_candidates.sort(key=lambda x: x[0])
        
        # Evaluate top K candidates exactly
        K = 20
        best_cand = None
        best_new_diam = float('inf')
        
        for new_d_st, u, v in scored_candidates[:K]:
            if time.time() > deadline:
                break
                
            new_edges = current_edges | {(u, v)}
            new_diam, _ = compute_diameter_and_pair(new_edges)
            
            if new_diam < best_new_diam:
                best_new_diam = new_diam
                best_cand = (u, v)
                
                if new_diam <= 4:
                    break
        
        if best_cand is None:
            # If no improvement in top K, try next batch or break
            # To ensure progress, we might need to look further
            # But if top 20 don't improve diameter, likely none will.
            # However, sometimes a "worse" local diameter reduction leads to better global structure?
            # Unlikely for this metric.
            break
            
        current_edges.add(best_cand)
        added.append(list(best_cand))
        
    return added
