import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set(tuple(e) for e in P['edges'])
    base_edges.update((v, u) for u, v in base_edges)
    
    # Deadline guard
    start_time = time.time()
    deadline = 50.0 # 10s margin
    
    # State
    added_edges = []
    added_set = set()
    
    def get_graph(added):
        adj = [[] for _ in range(n)]
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in added:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def bfs_dist(adj, src):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    def get_diameter(adj):
        max_d = 0
        worst_pair = (-1, -1)
        for i in range(n):
            dists = bfs_dist(adj, i)
            for j in range(i+1, n):
                if dists[j] > max_d:
                    max_d = dists[j]
                    worst_pair = (i, j)
        return max_d, worst_pair

    def is_valid(added):
        # Check triangle-free (bipartite)
        # Since base is bipartite, we just need to ensure added edges are bipartite
        # and don't create odd cycles.
        # Actually, if we only add edges between opposite parity, it stays bipartite.
        for u, v in added:
            if u % 2 == v % 2:
                return False
        # Check diameter
        adj = get_graph(added)
        d, _ = get_diameter(adj)
        return d <= 4

    # Greedy Phase
    while True:
        if time.time() > deadline:
            break
        adj = get_graph(added_edges)
        d, (u, v) = get_diameter(adj)
        if d <= 4:
            break
        
        # Find best edge to add
        best_edge = None
        best_score = float('inf')
        
        # Candidates: edges (a, b) with a < b, a % 2 != b % 2, not in base, not in added
        # Heuristic: prioritize edges that reduce dist(u, v)
        # dist(u, v) via (a, b) is min(dist(u,a)+1+dist(b,v), dist(u,b)+1+dist(a,v))
        # We need dist arrays from u and v? Or just use current dist?
        # Using current dist is an approximation.
        
        dist_u = bfs_dist(adj, u)
        dist_v = bfs_dist(adj, v)
        
        for a in range(n):
            for b in range(a+1, n):
                if a % 2 == b % 2:
                    continue
                if (a, b) in base_edges or (b, a) in base_edges:
                    continue
                if (a, b) in added_set:
                    continue
                
                # Estimate new distance
                d1 = dist_u[a] + 1 + dist_v[b]
                d2 = dist_u[b] + 1 + dist_v[a]
                new_d = min(d1, d2)
                
                # Prefer edges that make new_d small
                if new_d < best_score:
                    best_score = new_d
                    best_edge = (a, b)
                    
        if best_edge is None:
            break
            
        added_edges.append(list(best_edge))
        added_set.add(best_edge)

    # Local Search Phase (Edge Removal)
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        for i in range(len(added_edges) - 1, -1, -1):
            e = added_edges[i]
            # Try removing e
            added_edges.pop(i)
            added_set.remove(tuple(e))
            
            if is_valid(added_edges):
                removed_any = True
                # Keep it removed
            else:
                # Restore
                added_edges.insert(i, e)
                added_set.add(tuple(e))

    # Convert to list of lists
    result = [[u, v] for u, v in added_edges]
    return result
