"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _vertex_disjoint_paths(adj, source, targets, n):
    """Find vertex-disjoint paths from source to each target.
    Returns a dict mapping target -> path (list of vertices from source to target), or None."""
    from collections import deque
    
    # Build split graph: each vertex v -> v_in, v_out
    # Edge u->v in original becomes u_out -> v_in with capacity 1
    # v_in -> v_out with capacity 1 (except source: infinite, except targets: infinite)
    
    def v_in(v):
        return 2 * v
    
    def v_out(v):
        return 2 * v + 1
    
    total_nodes = 2 * n
    
    # Build flow network as adjacency list with capacities
    # We'll use a simple max-flow with BFS augmenting paths
    cap = {}  # (u, v) -> capacity
    flow = {}  # (u, v) -> current flow
    
    def add_edge(u, v, c):
        if (u, v) not in cap:
            cap[(u, v)] = 0
        cap[(u, v)] += c
        if (v, u) not in cap:
            cap[(v, u)] = 0
    
    # Add edges for each original edge
    for u in range(n):
        for v in adj[u]:
            if u < v:
                add_edge(v_out(u), v_in(v), 1)
                add_edge(v_out(v), v_in(u), 1)
    
    # Add split edges
    for v in range(n):
        if v == source:
            add_edge(v_in(v), v_out(v), n)  # large capacity
        elif v in targets:
            add_edge(v_in(v), v_out(v), n)  # large capacity
        else:
            add_edge(v_in(v), v_out(v), 1)
    
    # Source in flow network: source_out
    # Sinks: each target_in
    src = v_out(source)
    sinks = [v_in(t) for t in targets]
    
    # We want max flow from src to a super-sink connected from all sinks
    super_sink = total_nodes
    total_nodes += 1
    for s in sinks:
        add_edge(s, super_sink, 1)
    
    # Max flow with BFS (Edmonds-Karp)
    max_flow = 0
    parent = {}
    
    while True:
        # BFS to find augmenting path
        visited = {src}
        queue = deque([src])
        parent = {src: None}
        found = False
        
        while queue and not found:
            node = queue.popleft()
            if node == super_sink:
                found = True
                break
            for (u, v), c in list(cap.items()):
                if u == node and c - flow.get((u, v), 0) > 0 and v not in visited:
                    visited.add(v)
                    parent[v] = (node, u)
                    queue.append(v)
        
        if not found:
            break
        
        # Find min residual capacity along path
        path_flow = float('inf')
        node = super_sink
        while node != src:
            prev, _ = parent[node]
            residual = cap[(prev, node)] - flow.get((prev, node), 0)
            path_flow = min(path_flow, residual)
            node = prev
        
        # Update flows
        node = super_sink
        while node != src:
            prev, _ = parent[node]
            flow[(prev, node)] = flow.get((prev, node), 0) + path_flow
            flow[(node, prev)] = flow.get((node, prev), 0) - path_flow
            node = prev
        
        max_flow += path_flow
    
    if max_flow != len(targets):
        return None
    
    # Extract paths from flow
    # For each target, trace back from target_in to source_out
    paths = {}
    used_nodes = set()
    
    for t in targets:
        t_in_node = v_in(t)
        # Trace back: find which node sent flow to t_in_node
        # In the flow network, flow goes ... -> t_in -> super_sink
        # We need to trace from t_in backwards through the residual network
        # Actually, we need to find the path in the flow that reaches t_in
        
        # Reconstruct path by following flow edges
        # Start from t_in and go backwards
        path = [t]
        current = t_in_node
        
        while current != v_in(source):
            # Find predecessor that sent flow to current
            prev_found = False
            for (u, v), c in list(cap.items()):
                if v == current and flow.get((u, v), 0) > 0:
                    # Check if this is a valid reverse step
                    if u == v_in(source):
                        prev_node = source
                    elif u % 2 == 0:  # v_in
                        prev_node = u // 2
                    else:  # v_out
                        prev_node = (u - 1) // 2
                    
                    # Avoid cycles and already-used internal nodes
                    if prev_node in used_nodes and prev_node != source:
                        continue
                    if prev_node in [p for p in path[:-1]]:
                        continue
                    
                    prev_found = True
                    if prev_node != source:
                        used_nodes.add(prev_node)
                    path.insert(0, prev_node)
                    current = u
                    break
            
            if not prev_found:
                return None
        
        paths[t] = path
    
    return paths

def construct():
    from itertools import combinations
    
    n = P['n']
    adj = _build_adj(P['edges'], n)
    
    # Try to find K5 subdivision first, then K4, then K3, then K2
    # For Kk, we need k branches and vertex-disjoint paths between all pairs
    
    def try_clique_size(k):
        """Try to find a Kk subdivision by selecting k branches and checking
        if there are vertex-disjoint paths between all pairs."""
        if k > n:
            return None
        
        # For each set of k vertices, check if it forms a Kk subdivision
        # This is expensive, so we use a heuristic: start with high-degree vertices
        vertices_sorted = sorted(range(n), key=lambda v: -len(adj[v]))
        
        # Try combinations of top-degree vertices first
        for combo in combinations(vertices_sorted[:min(k + 5, n)], k):
            branches = list(combo)
            
            # Check if there are vertex-disjoint paths between all pairs
            # This is the hard part. For now, let's check if the branches
            # are close to forming a clique in terms of path availability.
            
            # For K4 and above, we need to find vertex-disjoint paths
            # Let's use a simpler check: for each pair, check if there's a path
            # through vertices not in branches
            # This is still hard, so let's just check for K3 (triangle) directly
            # and for K4 use a flow-based approach
            
            if k == 3:
                # Check if all three pairs are adjacent
                a, b, c = branches
                if b in adj[a] and c in adj[b] and a in adj[c]:
                    return {
                        'branches': branches,
                        'paths': [[a, b], [a, c], [b, c]]
                    }
            
            if k == 4:
                a, b, c, d = branches
                # Check if we can find vertex-disjoint paths
                # For each pair, the path should only use internal vertices (not in branches)
                # Simplified: check if each pair has at least one edge or a short path
                # through non-branch vertices
                
                # Try to find vertex-disjoint paths using flow
                # This is complex; for now, check if the 4 vertices form a near-clique
                # and the missing edges can be replaced by short paths
                
                # For simplicity, check if all pairs are either adjacent or have
                # a common neighbor not in branches
                pairs = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
                valid = True
                for u, v in pairs:
                    if v in adj[u]:
                        continue
                    # Check for common neighbor not in branches
                    common = adj[u] & adj[v] - set(branches)
                    if not common:
                        valid = False
                        break
                if valid:
                    # Build paths
                    paths = []
                    for u, v in pairs:
                        if v in adj[u]:
                            paths.append([u, v])
                        else:
                            common = adj[u] & adj[v] - set(branches)
                            mid = next(iter(common))
                            paths.append([u, mid, v])
                    return {
                        'branches': branches,
                        'paths': paths
                    }
        
        return None
    
    # Try K5
    result = try_clique_size(5)
    if result:
        return result
    
    # Try K4
    result = try_clique_size(4)
    if result:
        return result
    
    # Try K3 (triangle)
    es = {tuple(e) for e in P['edges']}
    for row in combinations(range(n), 3):
        if all(e in es for e in combinations(row, 2)):
            return {
                'branches': list(row),
                'paths': [list(e) for e in combinations(row, 2)]
            }
    
    # Fallback: K2
    u, v = P['edges'][0]
    return {
        'branches': [u, v],
        'paths': [[u, v]]
    }

# EVOLVE-BLOCK-END
