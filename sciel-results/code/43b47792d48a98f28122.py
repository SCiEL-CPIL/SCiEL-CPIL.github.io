"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _check_lift_fast(graph, signs):
    """
    Fast local check: build the lift adjacency, verify 3-regular and connected,
    return spectral radius (max abs eigenvalue of signed base adjacency).
    Returns (radius, details) or raises ValueError.
    """
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    e = len(edges)
    
    # Build signed adjacency matrix for base graph (n x n)
    A = np.zeros((n, n), dtype=np.float64)
    for idx, (u, v) in enumerate(edges):
        s = signs[idx]
        A[u, v] += s
        A[v, u] += s
    
    # Check 3-regular: each vertex should have degree 3 in the base graph
    # For prism graphs, this is guaranteed by structure, but let's verify
    # Actually, the lift is 3-regular iff the base is 3-regular (which it is for prisms)
    
    # Build lift graph for connectivity check
    # Lift has 2*n vertices: (v, 0) and (v, 1)
    # Edge (u,v) with sign +1: (u,0)-(v,0) and (u,1)-(v,1)
    # Edge (u,v) with sign -1: (u,0)-(v,1) and (u,1)-(v,0)
    lift_n = 2 * n
    lift_adj = [[] for _ in range(lift_n)]
    
    for idx, (u, v) in enumerate(edges):
        s = signs[idx]
        if s == 1:
            lift_adj[u * 2].append(v * 2)
            lift_adj[v * 2].append(u * 2)
            lift_adj[u * 2 + 1].append(v * 2 + 1)
            lift_adj[v * 2 + 1].append(u * 2 + 1)
        else:  # s == -1
            lift_adj[u * 2].append(v * 2 + 1)
            lift_adj[v * 2 + 1].append(u * 2)
            lift_adj[u * 2 + 1].append(v * 2)
            lift_adj[v * 2].append(u * 2 + 1)
    
    # Verify 3-regular
    for i in range(lift_n):
        if len(lift_adj[i]) != 3:
            raise ValueError(f"Vertex {i} has degree {len(lift_adj[i])}, not 3")
    
    # BFS for connectivity
    visited = [False] * lift_n
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        node = queue.pop(0)
        for neighbor in lift_adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                queue.append(neighbor)
    
    if count != lift_n:
        raise ValueError(f"Lift is not connected: only {count}/{lift_n} vertices reachable")
    
    # Compute spectral radius of signed base adjacency matrix
    eigvals = np.linalg.eigvalsh(A)
    radius = np.max(np.abs(eigvals))
    
    return radius, {"eigvals": eigvals}

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin within 60s limit
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    e = len(edges)
    
    # For prism graphs, the structure is:
    # Two m-cycles: vertices 0..m-1 and m..2m-1
    # Matching edges: (i, i+m) for i in 0..m-1
    # Total edges: 2*m (cycle edges) + m (matching edges) = 3*m
    
    # Key insight: gauge fixing. We can fix the signs of a spanning tree to +1
    # without loss of generality (up to global sign flips on vertices).
    # The remaining free signs are on the non-tree edges (the "cycle edges" that close cycles).
    
    # Let's identify the structure:
    # Cycle 1: 0-1-2-...-(m-1)-0 (edges: (0,1), (1,2), ..., (m-2,m-1), (m-1,0))
    # Cycle 2: m-(m+1)-...-(2m-1)-m (edges: (m,m+1), ..., (2m-2,2m-1), (2m-1,m))
    # Matching: (0,m), (1,m+1), ..., (m-1,2m-1)
    
    # We need to figure out which edges are in the graph['edges'] list.
    # Let's build a set of edges for quick lookup.
    edge_set = set()
    for (u, v) in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    # Identify cycle edges and matching edges
    # Cycle 1 edges: (i, i+1) for i in 0..m-2, and (m-1, 0)
    # Cycle 2 edges: (m+i, m+i+1) for i in 0..m-2, and (2m-1, m)
    # Matching edges: (i, m+i) for i in 0..m-1
    
    cycle1_edges = []
    cycle2_edges = []
    matching_edges = []
    
    for i in range(m):
        u = i
        v = (i + 1) % m
        cycle1_edges.append((min(u,v), max(u,v)))
        
        u = m + i
        v = m + (i + 1) % m
        cycle2_edges.append((min(u,v), max(u,v)))
        
        u = i
        v = m + i
        matching_edges.append((min(u,v), max(u,v)))
    
    # Gauge fixing: fix all matching edges to +1, and fix one edge from each cycle to +1
    # This leaves (m-1) + (m-1) = 2*(m-1) free binary variables
    # For m=4: 6 free vars -> 64 possibilities
    # For m=5: 8 free vars -> 256 possibilities
    # For m=7: 12 free vars -> 4096 possibilities
    # For m=9: 16 free vars -> 65536 possibilities
    
    # Let's build the edge index mapping
    edge_index = {}
    for idx, (u, v) in enumerate(edges):
        edge_index[(min(u,v), max(u,v))] = idx
    
    # Determine which edges are free
    # Fix all matching edges to +1
    # Fix one edge from cycle 1 to +1 (say the edge (0,1) if it exists, otherwise another)
    # Fix one edge from cycle 2 to +1
    
    fixed_plus = set()
    free_edges = []
    
    # Fix all matching edges
    for e_idx in matching_edges:
        if e_idx in edge_index:
            fixed_plus.add(edge_index[e_idx])
    
    # Fix one edge from cycle 1
    if m >= 2:
        # Use edge (0,1) or (0,m-1)
        e1 = (0, 1) if m > 1 else (0, 0)
        e1 = (min(e1[0], e1[1]), max(e1[0], e1[1]))
        if e1 in edge_index:
            fixed_plus.add(edge_index[e1])
        else:
            # Try another edge
            for ce in cycle1_edges:
                if ce in edge_index:
                    fixed_plus.add(edge_index[ce])
                    break
    
    # Fix one edge from cycle 2
    if m >= 2:
        e2 = (m, m+1) if m > 1 else (m, m)
        e2 = (min(e2[0], e2[1]), max(e2[0], e2[1]))
        if e2 in edge_index:
            fixed_plus.add(edge_index[e2])
        else:
            for ce in cycle2_edges:
                if ce in edge_index:
                    fixed_plus.add(edge_index[ce])
                    break
    
    # All other edges are free
    for idx in range(e):
        if idx not in fixed_plus:
            free_edges.append(idx)
    
    n_free = len(free_edges)
    
    # Exhaustive search over all 2^n_free combinations
    best_radius = float('inf')
    best_signs = None
    
    # For efficiency, precompute the base adjacency with fixed edges
    A_fixed = np.zeros((n, n), dtype=np.float64)
    for idx in fixed_plus:
        u, v = edges[idx]
        A_fixed[u, v] += 1
        A_fixed[v, u] += 1
    
    # For each combination of free edge signs, compute the spectral radius
    # We can use the fact that A = A_fixed + sum over free edges of s_i * (e_ui*ev_i + e_vi*eu_i)
    # For small n_free, exhaustive search is feasible
    
    if n_free <= 20:
        # Exhaustive search
        total = 1 << n_free
        # Process in chunks to allow deadline checking
        chunk_size = min(total, 100000)
        
        for start in range(0, total, chunk_size):
            if time.time() > deadline:
                break
            end = min(start + chunk_size, total)
            for mask in range(start, end):
                # Build signs vector
                signs = np.ones(e, dtype=np.int8)
                for j, idx in enumerate(free_edges):
                    if (mask >> j) & 1:
                        signs[idx] = -1
                
                # Check if this is a valid assignment (we already fixed some to +1)
                # The fixed_plus edges are already +1 in signs
                # But we need to make sure we didn't accidentally set them
                # Since free_edges excludes fixed_plus, this is fine
                
                # Quick validity: check if lift is connected
                # For prism graphs with gauge fixing, most assignments should be valid
                # But let's do a fast check
                
                # Build full A
                A = A_fixed.copy()
                for j, idx in enumerate(free_edges):
                    if (mask >> j) & 1:
                        u, v = edges[idx]
                        A[u, v] -= 2  # Change from +1 to -1
                        A[v, u] -= 2
                
                # Compute spectral radius
                eigvals = np.linalg.eigvalsh(A)
                radius = np.max(np.abs(eigvals))
                
                if radius < best_radius:
                    # Verify connectivity
                    try:
                        _, _ = _check_lift_fast(graph, signs)
                        best_radius = radius
                        best_signs = signs.copy()
                    except ValueError:
                        pass
    else:
        # For larger n_free, use random sampling with local search
        rng = np.random.default_rng(42)
        
        # Start with random assignments
        best_signs = None
        best_radius = float('inf')
        
        # Try several random starts
        n_starts = min(100, max(10, (int)(50000 / (2 ** n_free))))
        
        for _ in range(n_starts):
            if time.time() > deadline:
                break
            
            signs = np.ones(e, dtype=np.int8)
            for idx in free_edges:
                signs[idx] = rng.choice([-1, 1])
            
            # Local search: flip one free edge at a time
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for idx in free_edges:
                    if time.time() > deadline:
                        break
                    
                    old_sign = signs[idx]
                    signs[idx] = -old_sign
                    
                    try:
                        radius, _ = _check_lift_fast(graph, signs)
                        if radius < best_radius:
                            best_radius = radius
                            best_signs = signs.copy()
                            improved = True
                        else:
                            signs[idx] = old_sign
                    except ValueError:
                        signs[idx] = old_sign
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(e, dtype=np.int8)
        try:
            _, _ = _check_lift_fast(graph, best_signs)
        except ValueError:
            # Try flipping one edge
            for idx in range(e):
                test_signs = best_signs.copy()
                test_signs[idx] = -1
                try:
                    _, _ = _check_lift_fast(graph, test_signs)
                    best_signs = test_signs
                    break
                except ValueError:
                    continue
    
    return best_signs
# EVOLVE-BLOCK-END
