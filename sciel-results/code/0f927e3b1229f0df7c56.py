from collections import deque

def _bfs_diameter(adj):
    """Return the diameter of the graph given by adjacency lists.
    Returns float('inf') if the graph is disconnected."""
    n = len(adj)
    diam = 0
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
        for d in dist:
            if d == -1:
                return float('inf')
            if d > diam:
                diam = d
    return diam

def _build_adj(edges, n):
    """Build adjacency lists from edge list."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _is_critical(adj, u, v, n):
    """Check if edge (u,v) is critical: removing it makes diameter > 2 or disconnected.
    We temporarily remove the edge and check diameter."""
    # Remove edge u-v from adj
    adj[u].remove(v)
    adj[v].remove(u)
    diam = _bfs_diameter(adj)
    # Restore edge
    adj[u].append(v)
    adj[v].append(u)
    return diam > 2 or diam == float('inf')

def construct():
    import time
    deadline = time.time() + 80  # safety margin before 90s kill
    
    n = 16
    half = 8
    
    # Start with K_{8,8} - known valid with 64 edges
    base_edges = []
    for u in range(half):
        for v in range(half, n):
            base_edges.append([u, v])
    
    current_edges = list(base_edges)
    adj = _build_adj(current_edges, n)
    
    # Candidate edges to try adding (intra-partition edges)
    candidates = []
    for i in range(half):
        for j in range(i + 1, half):
            candidates.append([i, j])
    for i in range(half):
        for j in range(i + 1, half):
            candidates.append([half + i, half + j])
    
    # Sort candidates by some heuristic - maybe by how many "shared neighbors" they have
    # In K_{8,8}, any two vertices in same partition share all 8 neighbors in other partition
    # So all candidates are equivalent initially. Let's just try them in order.
    
    best_edges = list(current_edges)
    
    def try_add_edge(edges, adj, u, v):
        """Try adding edge (u,v). Returns (success, new_edges, new_adj) or (False, None, None)."""
        # Check if edge already exists
        if v in adj[u]:
            return False, None, None
        
        # Add edge temporarily
        test_adj = [list(lst) for lst in adj]
        test_adj[u].append(v)
        test_adj[v].append(u)
        
        # Check diameter is still 2
        diam = _bfs_diameter(test_adj)
        if diam != 2:
            return False, None, None
        
        # Check all edges are critical
        # This is expensive, so do it carefully
        for i, edge in enumerate(edges):
            a, b = edge
            if not _is_critical(test_adj, a, b, n):
                return False, None, None
        
        # Also check the new edge is critical
        if not _is_critical(test_adj, u, v, n):
            return False, None, None
        
        # Success
        test_edges = list(edges)
        test_edges.append([u, v])
        return True, test_edges, test_adj
    
    for cand in candidates:
        if time.time() > deadline:
            break
        u, v = cand
        if v in adj[u]:
            continue
        
        success, new_edges, new_adj = try_add_edge(current_edges, adj, u, v)
        if success:
            current_edges = new_edges
            adj = new_adj
            best_edges = list(current_edges)
    
    # Return as list of [u, v] with Python ints
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    return result
