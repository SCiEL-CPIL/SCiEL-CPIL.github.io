"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(n, adj, src):
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _compute_diameter(n, adj):
    max_dist = 0
    for i in range(n):
        dists = _bfs_distances(n, adj, i)
        for d in dists:
            if d > max_dist:
                max_dist = d
    return max_dist

def construct():
    n = P['n']
    base_edges = P['edges']
    
    # Pre-compute base adjacency
    base_adj = [[] for _ in range(n)]
    base_set = set()
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
        base_set.add((min(u, v), max(u, v)))
        
    # Candidate edges: even-odd pairs not in base
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2) and (u, v) not in base_set:
                candidates.append((u, v))
                
    added = []
    added_set = set()
    
    # Greedy Set-Cover style approach with lookahead
    while True:
        # Build current adjacency
        adj = [list(base_adj[i]) for i in range(n)]
        for u, v in added:
            adj[u].append(v)
            adj[v].append(u)
            
        # Compute all-pairs distances
        dists = [[0]*n for _ in range(n)]
        for i in range(n):
            d = _bfs_distances(n, adj, i)
            dists[i] = d
            
        # Find violating pairs (dist > 4)
        violating_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i][j] > 4:
                    violating_pairs.append((i, j))
                    
        if not violating_pairs:
            break
            
        # Evaluate candidates
        scored_candidates = []
        for u, v in candidates:
            if (u, v) in added_set:
                continue
                
            # Count how many violating pairs are fixed
            fixed_count = 0
            for i, j in violating_pairs:
                d_iu = dists[i][u]
                d_iv = dists[i][v]
                d_vj = dists[v][j]
                d_uj = dists[u][j]
                
                new_d = float('inf')
                if d_iu != -1 and d_vj != -1:
                    new_d = min(new_d, d_iu + 1 + d_vj)
                if d_iv != -1 and d_uj != -1:
                    new_d = min(new_d, d_iv + 1 + d_uj)
                    
                if new_d <= 4:
                    fixed_count += 1
                    
            if fixed_count > 0:
                scored_candidates.append((fixed_count, u, v))
                
        if not scored_candidates:
            # No candidate fixes any violation? Break to avoid infinite loop.
            # This shouldn't happen for connected bipartite graphs.
            break
            
        # Sort by fixed_count descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Lookahead: check top 5 candidates for actual new diameter
        best_edge = None
        best_diam = float('inf')
        
        # Limit to top 5 to save time
        for count, u, v in scored_candidates[:5]:
            test_adj = [list(adj[i]) for i in range(n)]
            test_adj[u].append(v)
            test_adj[v].append(u)
            new_diam = _compute_diameter(n, test_adj)
            
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = (u, v)
                
        if best_edge is None:
            # Fallback to the one with max fixed count
            best_edge = (scored_candidates[0][1], scored_candidates[0][2])
            
        added.append(best_edge)
        added_set.add(best_edge)
        
    return [[u, v] for u, v in added]
# EVOLVE-BLOCK-END
