import time

def _validate_edges(edges, n=12):
    """
    Validate that the edges form a simple undirected graph on vertices 0..n-1,
    with min degree >= 3, girth >= 4 (triangle-free), and contains at least one cycle.
    Returns (is_valid, distinct_cycle_lengths).
    """
    edge_set = set()
    for e in edges:
        if len(e) != 2:
            return False, None
        u, v = int(e[0]), int(e[1])
        if not (0 <= u < n) or not (0 <= v < n):
            return False, None
        if u == v:
            return False, None
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False, None
        edge_set.add(key)
    
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        u, v = int(u), int(v)
        adj[u].add(v)
        adj[v].add(u)
    
    for i in range(n):
        if len(adj[i]) < 3:
            return False, None
    
    for u, v in edge_set:
        common = adj[u] & adj[v]
        if common:
            return False, None
    
    cycle_lengths = set()
    
    def dfs(start, node, path, visited_in_path):
        next_nodes = sorted(adj[node])
        for next_node in next_nodes:
            if next_node == start and len(path) >= 3:
                cycle_lengths.add(len(path))
            elif next_node > start and next_node not in visited_in_path:
                dfs(start, next_node, path + [next_node], visited_in_path | {next_node})
    
    for i in range(n):
        dfs(i, i, [i], {i})
    
    if not cycle_lengths:
        return False, None
        
    return True, cycle_lengths

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Precomputed optimal graph: two disjoint K_{3,3} components.
    # Component 1: vertices 0,1,2 connected to 3,4,5
    # Component 2: vertices 6,7,8 connected to 9,10,11
    # This gives min degree 3, girth 4, and cycle lengths {4, 6}.
    # Raw value = 2. Score = seed/2.

    edges = [
        [0, 3], [0, 4], [0, 5],
        [1, 3], [1, 4], [1, 5],
        [2, 3], [2, 4], [2, 5],
        [6, 9], [6, 10], [6, 11],
        [7, 9], [7, 10], [7, 11],
        [8, 9], [8, 10], [8, 11]
    ]

    # Quick sanity check (fast enough to not matter)
    valid, lengths = _validate_edges(edges)
    if not valid:
        # Should never happen, but return edges anyway
        pass

    return edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    valid, lengths = _validate_edges(e)
    print(f"Valid: {valid}, Lengths: {lengths}")
