import time
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Leave 15s margin
    
    n = P['n']
    base_edges = set(tuple(e) for e in P['edges'])
    base_edges.update((v, u) for u, v in base_edges)
    
    # Helper to check triangle-free
    def is_triangle_free(graph_edges, new_edge):
        u, v = new_edge
        # Check if any common neighbor exists
        # Neighbors of u in current graph + new edge
        # Actually, just check if there is w such that (u,w) and (v,w) are in graph
        # Since graph is small, we can just check all w
        for w in range(n):
            if w == u or w == v:
                continue
            if (u, w) in graph_edges and (v, w) in graph_edges:
                return False
        return True

    def get_diameter(graph_edges):
        # BFS from all nodes
        max_dist = 0
        adj = {i: [] for i in range(n)}
        for u, v in graph_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            if -1 in dist:
                return float('inf') # Not connected
            max_dist = max(max_dist, max(dist))
        return max_dist

    # Candidate edges: all possible non-base edges
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                all_candidates.append((u, v))
    
    # Sort candidates by potential utility (shorter path reduction?)
    # Prefer edges that connect far apart nodes?
    # Or just random?
    # Let's use a greedy approach:
    # 1. Start with base graph.
    # 2. Find max distance pair.
    # 3. Pick best edge to reduce it.
    
    current_edges = set(base_edges)
    added_edges = []
    
    # To speed up, we can pre-compute distances
    def compute_all_pairs_distances(edges):
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        dists = {}
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            dists[start] = dist
        return dists

    # Greedy construction
    while True:
        if time.time() > deadline:
            break
            
        dists = compute_all_pairs_distances(current_edges)
        max_d = 0
        max_pair = None
        for u in range(n):
            for v in range(u + 1, n):
                if dists[u][v] > max_d:
                    max_d = dists[u][v]
                    max_pair = (u, v)
                    
        if max_d <= 4:
            break
            
        # Find best edge to add
        best_edge = None
        best_score = float('inf')
        
        # Limit candidates to those that might help
        # Ideally, an edge (a,b) should reduce distance between u and v
        # d(u,v) <= d(u,a) + 1 + d(b,v) or d(u,b) + 1 + d(a,v)
        
        for e in all_candidates:
            if e in current_edges:
                continue
            if not is_triangle_free(current_edges, e):
                continue
                
            a, b = e
            # Estimate new distance between max_pair
            u, v = max_pair
            d1 = dists[u][a] + 1 + dists[b][v]
            d2 = dists[u][b] + 1 + dists[a][v]
            new_d = min(d1, d2, dists[u][v])
            
            # We want to minimize the new diameter, but approximating with max_pair reduction
            # is good.
            if new_d < best_score:
                best_score = new_d
                best_edge = e
                
        if best_edge is None:
            # No valid edge found? Should not happen for n=24
            break
            
        current_edges.add(best_edge)
        added_edges.append(list(best_edge))
        
    # Try to remove edges (Local Search)
    # Iterate through added edges, try to remove each if valid
    for i in range(len(added_edges)):
        if time.time() > deadline:
            break
        e = added_edges[i]
        # Remove e
        current_edges.discard(e)
        
        # Check validity
        if is_triangle_free(current_edges, e): # This check is redundant if we just removed it, but we need to check the graph is still triangle free (it is, as we removed an edge)
             # Wait, removing an edge cannot create a triangle.
             pass
             
        dists = compute_all_pairs_distances(current_edges)
        max_d = max(max(d) for d in dists.values())
        
        if max_d <= 4:
            # Keep it removed
            added_edges.pop(i)
            i -= 1 # Re-check same index
        else:
            # Restore
            current_edges.add(e)
            
    return [list(e) for e in added_edges]

