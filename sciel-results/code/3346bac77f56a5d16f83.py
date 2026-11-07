def sign_edges(graph, budget):
    import numpy as np
    import time
    
    deadline = time.time() + 55  # 55s deadline, 5s safety margin
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Build adjacency structure
    # signed adjacency matrix A where A[u,v] = sign for edge (u,v)
    
    def build_matrix(signs):
        A = np.zeros((n, n), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            A[u, v] = signs[idx]
            A[v, u] = signs[idx]
        return A
    
    def spectral_radius(A):
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_valid_lift(signs):
        """Check if the 2-lift is connected and 3-regular."""
        # 3-regular is guaranteed by the graph structure
        # Connectivity: the lift has 2n vertices, each with degree 3
        # The lift is connected iff the signs don't all form a "coboundary"
        # i.e., there's no vertex labeling such that all edge signs are products of vertex labels
        
        # Build the lift graph and check connectivity
        # Vertices in lift: (v, 0) and (v, 1) for v in 0..n-1
        # Edge (u,v) with sign +1: connects (u,0)-(v,0) and (u,1)-(v,1)
        # Edge (u,v) with sign -1: connects (u,0)-(v,1) and (u,1)-(v,0)
        
        from collections import deque
        
        lift_n = 2 * n
        adj = [[] for _ in range(lift_n)]
        
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                adj[u * 2].append(v * 2)
                adj[v * 2].append(u * 2)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:  # s == -1
                adj[u * 2].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2)
                adj[u * 2 + 1].append(v * 2)
                adj[v * 2].append(u * 2 + 1)
        
        # BFS from vertex 0
        visited = [False] * lift_n
        queue = deque([0])
        visited[0] = True
        count = 1
        
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        
        return count == lift_n
    
    # Initial signs: try structured patterns first
    # Pattern 1: all +1 (may not be connected)
    # Pattern 2: alternating signs on cycles
    
    best_signs = None
    best_radius = float('inf')
    
    # Generate several initial candidates
    candidates = []
    
    # All +1
    candidates.append(np.ones(num_edges, dtype=np.int8))
    
    # Alternating on top cycle, +1 on matching, alternating on bottom
    signs_alt = np.ones(num_edges, dtype=np.int8)
    # Edges are ordered: let's assume first m are top cycle, next m bottom cycle, last m matching
    # But we need to be careful about edge ordering
    
    # Let's just do random initialization with the given seed
    rng = np.random.default_rng(3)
    
    # Try the seed's approach first
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=num_edges)
        if is_valid_lift(signs):
            A = build_matrix(signs)
            r = spectral_radius(A)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: find any valid lift
        signs = np.ones(num_edges, dtype=np.int8)
        # Make it valid by setting one edge to -1 if needed
        if not is_valid_lift(signs):
            signs[0] = -1
            if not is_valid_lift(signs):
                signs[1] = -1
        best_signs = signs
        A = build_matrix(signs)
        best_radius = spectral_radius(A)
    
    # Local search
    A = build_matrix(best_signs)
    current_radius = best_radius
    
    while time.time() < deadline:
        improved = False
        for idx in range(num_edges):
            if time.time() >= deadline:
                break
            
            # Try flipping sign at idx
            old_sign = best_signs[idx]
            best_signs[idx] = -old_sign
            
            if not is_valid_lift(best_signs):
                best_signs[idx] = old_sign
                continue
            
            A_new = build_matrix(best_signs)
            r_new = spectral_radius(A_new)
            
            if r_new < current_radius - 1e-10:
                current_radius = r_new
                best_radius = r_new
                improved = True
            else:
                best_signs[idx] = old_sign
        
        if not improved:
            # Try random restarts or perturbation
            # Perturb a few random edges
            for _ in range(5):
                if time.time() >= deadline:
                    break
                idx = rng.integers(0, num_edges)
                old_sign = best_signs[idx]
                best_signs[idx] = -old_sign
                if is_valid_lift(best_signs):
                    A_new = build_matrix(best_signs)
                    r_new = spectral_radius(A_new)
                    if r_new < current_radius:
                        current_radius = r_new
                        best_radius = r_new
                        improved = True
                    else:
                        best_signs[idx] = old_sign
            if not improved:
                break
    
    return best_signs.astype(np.int8)
