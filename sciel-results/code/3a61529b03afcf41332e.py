def sign_edges(graph, budget):
    import numpy as np
    import time
    
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    # Precompute edge indices
    edge_u = np.array([e[0] for e in edges], dtype=int)
    edge_v = np.array([e[1] for e in edges], dtype=int)

    # Identify edge types to help build a specific spanning tree
    # Top cycle: 0..m-1
    # Bot cycle: m..2m-1
    # Matching: i <-> i+m
    
    # We will construct a spanning tree consisting of:
    # 1. All matching edges (i, i+m) for i in 0..m-1
    # 2. All top cycle edges (i, i+1) for i in 0..m-2 (and (m-1, 0) is NOT in tree? 
    #    Let's use a path on top: (0,1), (1,2), ..., (m-2, m-1).
    #    This connects all top vertices to 0, and all bot vertices to their matching top vertex.
    #    So the tree has:
    #    - m matching edges
    #    - m-1 top edges
    #    Total m + m - 1 = 2m - 1 edges. This is a spanning tree for 2m vertices.
    
    tree_edge_indices = set()
    non_tree_edge_indices = []
    
    # Map edge (u,v) to index
    edge_map = {}
    for idx, (u, v) in enumerate(edges):
        edge_map[(u, v)] = idx
        edge_map[(v, u)] = idx

    # Add matching edges to tree
    for i in range(m):
        idx = edge_map.get((i, i + m))
        if idx is not None:
            tree_edge_indices.add(idx)
    
    # Add top path edges to tree
    for i in range(m - 1):
        idx = edge_map.get((i, i + 1))
        if idx is not None:
            tree_edge_indices.add(idx)
        # Handle wrap-around if the graph edges are ordered differently?
        # The problem says "two m-cycles". Usually edges are (i, i+1) mod m.
        # If (m-1, 0) is an edge, it is NOT in the tree.
        
    # Identify non-tree edges
    for idx in range(n_edges):
        if idx not in tree_edge_indices:
            non_tree_edge_indices.append(idx)
            
    # The number of non-tree edges should be m+1
    # Let's verify: Total edges 3m. Tree edges 2m-1. Non-tree = 3m - (2m-1) = m+1. Correct.
    
    k = len(non_tree_edge_indices)
    if k != m + 1:
        # Fallback to random if structure is unexpected
        # But prism graphs are standard.
        pass

    best_signs = None
    best_radius = float('inf')
    
    # Enumerate all 2^k switching classes (cycle product patterns)
    # We represent the pattern as an integer from 0 to 2^k - 1
    # Bit i of the integer corresponds to the sign of the i-th non-tree edge
    # 0 -> +1, 1 -> -1
    
    for mask in range(1 << k):
        if (time.time() - start_time) > DEADLINE:
            break
            
        # Construct signs
        signs = np.ones(n_edges, dtype=int)
        
        # Non-tree edges get signs from the mask
        for j, e_idx in enumerate(non_tree_edge_indices):
            if (mask >> j) & 1:
                signs[e_idx] = -1
            else:
                signs[e_idx] = 1
                
        # Tree edges are +1 (already set by np.ones)
        
        # Check if balanced (all cycle products +1)
        # This happens if all non-tree edge signs are +1 (since tree edges are +1)
        # Wait, the cycle product is the product of signs on the cycle.
        # For fundamental cycle of non-tree edge e_j, the product is s_{e_j} * (prod tree signs) = s_{e_j} * 1 = s_{e_j}.
        # So if all s_{e_j} = 1, then all fundamental cycle products are 1.
        # Does this imply all cycle products are 1? Yes, because fundamental cycles form a basis.
        # So the graph is balanced iff mask == 0.
        if mask == 0:
            continue
            
        # Build signed adjacency matrix
        A = np.zeros((n_vertices, n_vertices))
        for idx in range(n_edges):
            s = signs[idx]
            u, v = edge_u[idx], edge_v[idx]
            A[u, v] = s
            A[v, u] = s
            
        # Compute spectral radius
        # A is symmetric
        vals = np.linalg.eigvalsh(A)
        radius = float(np.max(np.abs(vals)))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    # If no valid signs found (should not happen for m>=3), fallback
    if best_signs is None:
        # Try a random valid sign
        rng = np.random.default_rng(42)
        for _ in range(1000):
            signs = rng.choice([-1, 1], size=n_edges).astype(int)
            # Check balance
            # Simple check: BFS
            sigma = np.zeros(n_vertices, dtype=int)
            sigma[0] = 1
            visited = [False] * n_vertices
            visited[0] = True
            q = [0]
            balanced = True
            while q:
                u = q.pop(0)
                for idx in range(n_edges):
                    if edge_u[idx] == u:
                        v = edge_v[idx]
                        s = signs[idx]
                        exp_sig = s * sigma[u]
                        if not visited[v]:
                            sigma[v] = exp_sig
                            visited[v] = True
                            q.append(v)
                        else:
                            if sigma[v] != exp_sig:
                                balanced = False
                                break
                if not balanced:
                    break
            if not balanced:
                A = np.zeros((n_vertices, n_vertices))
                for idx in range(n_edges):
                    s = signs[idx]
                    u, v = edge_u[idx], edge_v[idx]
                    A[u, v] = s
                    A[v, u] = s
                vals = np.linalg.eigvalsh(A)
                radius = float(np.max(np.abs(vals)))
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs
            if best_signs is not None:
                break

    return best_signs
