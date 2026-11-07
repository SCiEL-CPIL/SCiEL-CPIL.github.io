"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(n, adj, added_edges):
    # Check if adding any edge in added_edges creates a triangle
    # We check the full graph
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return False
    return True

def _compute_diameter(n, adj):
    # BFS from each node to find max shortest path
    from collections import deque
    max_diam = 0
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
        max_diam = max(max_diam, max(dist))
    return max_diam

def _can_add_edge(n, adj, u, v):
    # Check if adding edge (u,v) keeps graph triangle-free
    # A triangle would be (u, v, w) where w is common neighbor of u and v
    common = adj[u] & adj[v]
    return len(common) == 0

def _all_pairs_shortest(n, adj):
    from collections import deque
    dist = [[-1] * n for _ in range(n)]
    for start in range(n):
        dist[start][start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[start][v] == -1:
                    dist[start][v] = dist[start][u] + 1
                    q.append(v)
    return dist

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u,v), max(u,v)))
    
    adj = _build_adj(n, P['edges'])
    
    # Greedy: add edges to reduce diameter
    added = []
    added_set = set()
    
    while time.time() < deadline:
        diam = _compute_diameter(n, adj)
        if diam <= 4:
            break
        
        # Find the pair with max distance
        dist = _all_pairs_shortest(n, adj)
        max_dist = 0
        max_pair = None
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] > max_dist:
                    max_dist = dist[i][j]
                    max_pair = (i, j)
        
        if max_pair is None:
            break
        
        # Try to add an edge that reduces the diameter
        # Candidate edges: all pairs not already in graph, that are triangle-free
        best_edge = None
        best_diam_after = max_dist + 1
        
        u, v = max_pair
        # Try adding edge (u,v) directly
        if (min(u,v), max(u,v)) not in base_edges and (min(u,v), max(u,v)) not in added_set:
            if _can_add_edge(n, adj, u, v):
                # Temporarily add
                adj[u].add(v)
                adj[v].add(u)
                d = _compute_diameter(n, adj)
                adj[u].remove(v)
                adj[v].remove(u)
                if d < best_diam_after:
                    best_diam_after = d
                    best_edge = (u, v)
        
        # Also try edges that might help: connect u to something closer to v, or v to something closer to u
        # Try all candidate edges that are triangle-free
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) in base_edges or (i, j) in added_set:
                    continue
                if not _can_add_edge(n, adj, i, j):
                    continue
                # Check if this edge helps reduce max distance
                # Simple heuristic: does it reduce dist between some far pair?
                adj[i].add(j)
                adj[j].add(i)
                d = _compute_diameter(n, adj)
                adj[i].remove(j)
                adj[j].remove(i)
                if d < best_diam_after:
                    best_diam_after = d
                    best_edge = (i, j)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added.append([u, v])
        added_set.add((min(u,v), max(u,v)))
    
    # Post-hoc: try to remove edges to minimize count
    if time.time() < deadline:
        for idx in range(len(added) - 1, -1, -1):
            if time.time() > deadline:
                break
            u, v = added[idx]
            # Remove edge
            adj[u].remove(v)
            adj[v].remove(u)
            # Check constraints
            if _is_triangle_free(n, adj, []):
                diam = _compute_diameter(n, adj)
                if diam <= 4:
                    # Keep removed
                    added.pop(idx)
                    added_set.discard((min(u,v), max(u,v)))
                else:
                    # Restore
                    adj[u].add(v)
                    adj[v].add(u)
            else:
                # Restore
                adj[u].add(v)
                adj[v].add(u)
    
    # Ensure types are correct
    result = [[int(u), int(v)] for u, v in added]
    return result

# EVOLVE-BLOCK-END
