"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import time
    import numpy as np
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin

    def build_luw(q, d):
        """Build LUW(d) graph over GF(q). Returns (n, edges)."""
        # Precompute modular arithmetic tables for speed
        # Define the recurrence rule
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        # Generate all vertices
        n = 2 * q ** d
        qd = q ** d
        
        # Map vertex to index
        def vertex_to_index(side, x_tuple):
            idx = 0
            for i in range(d):
                idx = idx * q + x_tuple[i]
            return side * qd + idx
        
        edges_list = []
        seen_edges = set()
        
        for side in [0]:  # Only generate from side 0 to avoid duplicates
            for idx in range(qd):
                # Decode idx to tuple
                x = []
                temp = idx
                for i in range(d):
                    x.append(temp % q)
                    temp //= q
                x = tuple(x)
                
                # For each "first" value in GF(q), compute the neighbor
                for first in range(q):
                    # y[0] is "None" (not used), y[1] = first, y[2..d] computed
                    y = [None] + [0] * d  # y[0] unused, y[1..d]
                    y[1] = first
                    for c in range(2, d + 1):
                        a, b = rule[c]
                        if side == 0:
                            y[c] = (x[c-1] + y[a] * x[b-1]) % q
                        else:
                            y[c] = (x[c-1] - x[a-1] * y[b]) % q
                    neighbor_idx = vertex_to_index(1 - side, tuple(y[1:]))
                    v_idx = vertex_to_index(side, x)
                    edge = (v_idx, neighbor_idx)
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        edges_list.append(edge)
        
        return n, edges_list

    def verify_girth_nonbacktracking(n, edges, target_girth):
        """Verify girth >= target_girth using non-backtracking walks."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Build directed edge list
        dir_edges = []
        dir_edge_idx = {}
        for u in range(n):
            for v in adj[u]:
                key = (u, v)
                if key not in dir_edge_idx:
                    dir_edge_idx[key] = len(dir_edges)
                    dir_edges.append((u, v))
        
        num_dir_edges = len(dir_edges)
        
        # Build non-backtracking transition
        nb_trans = [[] for _ in range(num_dir_edges)]
        for (u, v), idx in dir_edge_idx.items():
            for w in adj[v]:
                if w != u:
                    next_key = (v, w)
                    if next_key in dir_edge_idx:
                        nb_trans[idx].append(dir_edge_idx[next_key])
        
        # For each L from 2 to target_girth-1, check if any non-backtracking
        # closed walk of length L exists.
        for L in range(2, target_girth):
            # Initialize: edge_reachable[edge_idx] = {start_vertex_of_edge}
            edge_reachable = {}
            for (u, v), idx in dir_edge_idx.items():
                edge_reachable[idx] = {u}
            
            # Iterate L-1 times
            for step in range(L - 1):
                new_reachable = {}
                for e_idx, starts in edge_reachable.items():
                    for f_idx in nb_trans[e_idx]:
                        if f_idx not in new_reachable:
                            new_reachable[f_idx] = set()
                        new_reachable[f_idx].update(starts)
                edge_reachable = new_reachable
            
            # Check if any edge (c, a) has a in its reachable set
            for (u, v), idx in dir_edge_idx.items():
                # Edge (u,v): end vertex is v. Check if v is in edge_reachable[idx]
                if idx in edge_reachable and v in edge_reachable[idx]:
                    return False  # Found a cycle of length L
        
        return True

    # For k=5, g=9:
    # LUW(d) over GF(q) has degree q and girth >= 2d+2
    # We need degree k, so q = k = 5
    # We need girth >= g, so 2d+2 >= g, i.e., d >= (g-2)/2
    # For g=9: d >= 3.5, so d >= 4
    # LUW(4) over GF(5): n = 2*5^4 = 1250
    
    q = k
    min_d = max(2, (g - 2) // 2 + 1)  # d such that 2d+2 >= g
    
    best_result = None
    
    # Try d values starting from minimum
    for d in range(min_d, min_d + 5):
        if time.time() > deadline - 20:
            break
        try:
            n, edges = build_luw(q, d)
            # Verify the graph is k-regular
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(deg[i] != k for i in range(n)):
                continue
            # Verify girth
            if verify_girth_nonbacktracking(n, edges, g):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
        except Exception as e:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: try larger d
    for d in range(min_d + 5, min_d + 8):
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw(q, d)
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(deg[i] != k for i in range(n)):
                continue
            if verify_girth_nonbacktracking(n, edges, g):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    # Last resort
    d = min_d + 2
    n, edges = build_luw(q, d)
    return n, edges
# EVOLVE-BLOCK-END
