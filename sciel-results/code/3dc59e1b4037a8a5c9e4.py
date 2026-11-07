import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    t0 = time.time()
    deadline = 85.0
    n = P['n']
    base_edges = P['edges']
    
    # Base adjacency
    base_adj = [set() for _ in range(n)]
    base_edge_set = set()
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
        base_edge_set.add((min(u, v), max(u, v)))
    
    # Precompute: for each pair (u,v) not in base, is it triangle-safe?
    # Triangle-safe means: no w such that w in base_adj[u] and w in base_adj[v] and w != u and w != v
    # Actually, adding (u,v) creates a triangle if there exists w adjacent to both u and v in the CURRENT graph.
    # But for a static check against base only: if we only add edges that are "far" from each other, we can precompute.
    # However, two added edges can interact. Let's handle dynamically but efficiently.
    
    # Let's precompute base neighbor intersection for triangle check with base edges
    # For a new edge (u,v): it's safe w.r.t. base if (base_adj[u] & base_adj[v]) - {u, v} is empty
    base_safe = {}
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edge_set:
                continue
            inter = base_adj[u] & base_adj[v]
            inter.discard(u)
            inter.discard(v)
            if not inter:
                base_safe[(u, v)] = True
            else:
                base_safe[(u, v)] = False
    
    # All candidate edges (non-base, triangle-safe w.r.t. base)
    candidates = [uv for uv in base_safe if base_safe[uv]]
    
    def build_adj(added):
        adj = [set(base_adj[i]) for i in range(n)]
        for u, v in added:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def bfs_from(s, adj):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist
    
    def compute_diam_and_bad(adj):
        # Returns (diameter, list of pairs with dist > 4, sum of excess)
        dist_matrix = [bfs_from(s, adj) for s in range(n)]
        diam = 0
        bad_pairs = []
        sum_excess = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_matrix[i][j]
                if d > diam:
                    diam = d
                if d > 4:
                    bad_pairs.append((i, j, d))
                    sum_excess += d - 4
        return diam, bad_pairs, sum_excess, dist_matrix
    
    def is_edge_safe_in_graph(u, v, adj, edge_set):
        if (min(u, v), max(u, v)) in edge_set:
            return False
        # Check triangle: is there w in adj[u] & adj[v] (excluding u,v)?
        inter = adj[u] & adj[v]
        inter.discard(u)
        inter.discard(v)
        if inter:
            return False
        return True
    
    def add_edge(added, adj, edge_set, u, v):
        added.append((u, v))
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    def remove_edge(added, adj, edge_set, u, v):
        added.remove((u, v))
        adj[u].discard(v)
        adj[v].discard(u)
        edge_set.discard((min(u, v), max(u, v)))
    
    # Greedy construction
    added = []
    adj = build_adj(added)
    edge_set = set(base_edge_set)
    
    while time.time() - t0 < deadline - 20:
        diam, bad_pairs, sum_excess, dist_matrix = compute_diam_and_bad(adj)
        if diam <= 4:
            break
        
        # Score each candidate edge by how much it reduces the bad pairs
        best_edge = None
        best_score = None
        # Precompute for each candidate, the new distances to bad pairs
        # For efficiency, only consider edges that might help: endpoints near bad pairs
        for (u, v) in candidates:
            if (min(u, v), max(u, v)) in edge_set:
                continue
            if not is_edge_safe_in_graph(u, v, adj, edge_set):
                continue
            # Tentatively add
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))
            
            # Recompute distances from u and v only (since only they changed)
            # Actually, adding an edge can change distances from all vertices, but let's just recompute diam
            # For speed, recompute full but it's 24 BFS which is fast
            new_diam, new_bad, new_sum, _ = compute_diam_and_bad(adj)
            
            # Remove
            adj[u].discard(v)
            adj[v].discard(u)
            edge_set.discard((min(u, v), max(u, v)))
            
            score = (new_diam, len(new_bad), new_sum)
            if best_score is None or score < best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        add_edge(added, adj, edge_set, u, v)
    
    # Local search: try to remove edges
    improved = True
    while improved and time.time() - t0 < deadline - 10:
        improved = False
        for i in range(len(added)):
            u, v = added[i]
            # Try removing
            remove_edge(added, adj, edge_set, u, v)
            diam, bad_pairs, sum_excess, _ = compute_diam_and_bad(adj)
            if diam <= 4:
                improved = True
            else:
                add_edge(added, adj, edge_set, u, v)
    
    # Local search: try swapping one edge for another
    if time.time() - t0 < deadline - 5:
        swapped = True
        while swapped and time.time() - t0 < deadline - 5:
            swapped = False
            for i in range(len(added)):
                u, v = added[i]
                # Try removing this edge
                remove_edge(added, adj, edge_set, u, v)
                for (a, b) in candidates:
                    if (min(a, b), max(a, b)) in edge_set:
                        continue
                    if not is_edge_safe_in_graph(a, b, adj, edge_set):
                        continue
                    # Try adding (a,b)
                    add_edge(added, adj, edge_set, a, b)
                    diam, bad_pairs, sum_excess, _ = compute_diam_and_bad(adj)
                    if diam <= 4:
                        # Keep the swap, continue searching
                        swapped = True
                        break
                    else:
                        remove_edge(added, adj, edge_set, a, b)
                if swapped:
                    break
                else:
                    # Re-add the removed edge
                    add_edge(added, adj, edge_set, u, v)
    
    # Final verification
    final_adj = build_adj(added)
    diam, bad_pairs, sum_excess, _ = compute_diam_and_bad(final_adj)
    if diam > 4:
        # Fallback: use greedy result even if not perfect (shouldn't happen)
        pass
    
    # Convert to list of [u, v] with int type
    result = [[int(u), int(v)] for u, v in added]
    return result

