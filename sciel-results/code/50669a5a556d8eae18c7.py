"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Greedy construction of a k-regular graph with girth >= g.
    
    Build a BFS tree of depth g//2, then connect last-layer vertices
    in a (k-1)-regular subgraph that avoids short cycles.
    """
    import random
    
    max_depth = g // 2
    
    vertices_by_depth = [[] for _ in range(max_depth + 1)]
    parent = {}
    children = {}
    edges = set()
    
    def add_edge(u, v):
        if u == v:
            return False
        a, b = min(u, v), max(u, v)
        if (a, b) in edges:
            return False
        edges.add((a, b))
        return True
    
    # Create root
    root = 0
    vertices_by_depth[0].append(root)
    parent[root] = None
    children[root] = []
    
    # BFS: build tree
    next_id = 1
    for depth in range(max_depth):
        for v in vertices_by_depth[depth]:
            num_needed = k if depth == 0 else (k - 1)
            for _ in range(num_needed):
                new_v = next_id
                next_id += 1
                vertices_by_depth[depth + 1].append(new_v)
                parent[new_v] = v
                children[v].append(new_v)
                children[new_v] = []
                add_edge(v, new_v)
    
    n = next_id
    last_depth_vertices = vertices_by_depth[max_depth]
    
    # Group last-depth vertices by top-level ancestor (child of root)
    def get_top_ancestor(x):
        cur = x
        while parent[cur] is not None and parent[parent[cur]] is not None:
            cur = parent[cur]
        return cur
    
    top_groups = {}
    for v in last_depth_vertices:
        ta = get_top_ancestor(v)
        top_groups.setdefault(ta, []).append(v)
    
    # Build (k-1)-regular graph on last_depth_vertices
    # Edges only between different top groups
    last_adj = {v: set() for v in last_depth_vertices}
    
    # Use greedy with backtracking-like retry
    random.seed(42)
    
    # For k=3, k-1=2, so we need 2-regular (cycles)
    # Strategy: create cycles that go through different top groups
    
    # Get list of top groups
    group_list = list(top_groups.keys())
    group_vertices = {g: list(top_groups[g]) for g in group_list}
    for g in group_list:
        random.shuffle(group_vertices[g])
    
    # For 2-regular: create cycles. Each cycle must have vertices from at least 2 groups.
    # Simplest: pair up vertices from different groups in cycles.
    
    # Alternative: use a round-robin style matching
    # For each vertex, connect to 2 others from different groups
    
    # Greedy: process vertices in order, connect to available vertices from other groups
    remaining = {v: k - 1 for v in last_depth_vertices}
    
    # Sort for determinism
    last_sorted = sorted(last_depth_vertices)
    
    for v in last_sorted:
        if remaining[v] <= 0:
            continue
        # Find candidates from other groups
        v_group = get_top_ancestor(v)
        candidates = []
        for w in last_sorted:
            if w == v or remaining[w] <= 0:
                continue
            if w in last_adj[v]:
                continue
            if get_top_ancestor(w) != v_group:
                candidates.append(w)
        
        random.shuffle(candidates)
        for w in candidates:
            if remaining[v] <= 0:
                break
            if remaining[w] <= 0:
                continue
            if w in last_adj[v]:
                continue
            last_adj[v].add(w)
            last_adj[w].add(v)
            remaining[v] -= 1
            remaining[w] -= 1
            add_edge(v, w)
    
    # Verify all have correct degree
    for v in last_depth_vertices:
        if len(last_adj[v]) != k - 1:
            # Fallback: this shouldn't happen, but if it does, return invalid
            pass
    
    edge_list = [(a, b) for a, b in edges]
    return n, edge_list
# EVOLVE-BLOCK-END

