import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build edge index mapping for quick lookup
    # Edge types: cycle1 (0..m-1), cycle2 (m..2m-1), matching (2m..3m-1)
    # But we should use the actual edge list
    
    # Build adjacency structure for the base graph
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    def build_signed_adj(signs):
        """Build signed adjacency matrix."""
        A = np.zeros((n, n), dtype=float)
        for idx, (u, v) in enumerate(edges):
            A[u, v] = signs[idx]
            A[v, u] = signs[idx]
        return A
    
    def spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_lift_connected(signs):
        """Check if the 2-lift is connected.
        
        The 2-lift is connected iff there exists at least one cycle in the base
        graph with negative sign product. For prism graphs, we check the
        fundamental cycles.
        """
        # Cycle 1: top cycle 0-1-2-...-(m-1)-0
        prod1 = 1
        for i in range(m):
            u, v = i, (i + 1) % m
            # Find edge index
            for idx, (eu, ev) in enumerate(edges):
                if (eu == u and ev == v) or (eu == v and ev == u):
                    prod1 *= signs[idx]
                    break
        if prod1 == -1:
            return True
        
        # Cycle 2: bottom cycle m-(m+1)-...-(2m-1)-m
        prod2 = 1
        for i in range(m):
            u, v = m + i, m + (i + 1) % m
            for idx, (eu, ev) in enumerate(edges):
                if (eu == u and ev == v) or (eu == v and ev == u):
                    prod2 *= signs[idx]
                    break
        if prod2 == -1:
            return True
        
        # Ladder cycles: for each i, cycle (i, i+1, i+1+m, i+m, i)
        for i in range(m):
            prod = 1
            # Edge (i, i+1)
            u, v = i, (i + 1) % m
            for idx, (eu, ev) in enumerate(edges):
                if (eu == u and ev == v) or (eu == v and ev == u):
                    prod *= signs[idx]
                    break
            # Edge (i+1, i+1+m)
            u, v = (i + 1) % m, m + (i + 1) % m
            for idx, (eu, ev) in enumerate(edges):
                if (eu == u and ev == v) or (eu == v and ev == u):
                    prod *= signs[idx]
                    break
            # Edge (i+1+m, i+m)
            u, v = m + (i + 1) % m, m + i
            for idx, (eu, ev) in enumerate(edges):
                if (eu == u and ev == v) or (eu == v and ev == u):
                    prod *= signs[idx]
                    break
            # Edge (i+m, i)
            u, v = m + i, i
            for idx, (eu, ev) in enumerate(edges):
                if (eu == u and ev == v) or (eu == v and ev == u):
                    prod *= signs[idx]
                    break
            if prod == -1:
                return True
        
        return False
    
    # Optimize: precompute edge indices for cycles
    def find_edge_idx(u, v):
        for idx, (eu, ev) in enumerate(edges):
            if (eu == u and ev == v) or (eu == v and ev == u):
                return idx
        return -1
    
    # Precompute cycle edge indices
    top_cycle_edges = []
    for i in range(m):
        idx = find_edge_idx(i, (i + 1) % m)
        top_cycle_edges.append(idx)
    
    bottom_cycle_edges = []
    for i in range(m):
        idx = find_edge_idx(m + i, m + (i + 1) % m)
        bottom_cycle_edges.append(idx)
    
    ladder_cycle_edges = []
    for i in range(m):
        cycle = []
        u1, v1 = i, (i + 1) % m
        cycle.append(find_edge_idx(u1, v1))
        u2, v2 = (i + 1) % m, m + (i + 1) % m
        cycle.append(find_edge_idx(u2, v2))
        u3, v3 = m + (i + 1) % m, m + i
        cycle.append(find_edge_idx(u3, v3))
        u4, v4 = m + i, i
        cycle.append(find_edge_idx(u4, v4))
        ladder_cycle_edges.append(cycle)
    
    def is_lift_connected_fast(signs):
        """Fast connectivity check using precomputed cycle edges."""
        # Top cycle
        prod = 1
        for idx in top_cycle_edges:
            prod *= signs[idx]
        if prod == -1:
            return True
        
        # Bottom cycle
        prod = 1
        for idx in bottom_cycle_edges:
            prod *= signs[idx]
        if prod == -1:
            return True
        
        # Ladder cycles
        for cycle in ladder_cycle_edges:
            prod = 1
            for idx in cycle:
                prod *= signs[idx]
            if prod == -1:
                return True
        
        return False
    
    def local_search(signs, max_iters=10000):
        """Local search: flip signs to reduce spectral radius."""
        current = signs.copy()
        current_radius = spectral_radius(current)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try flipping each edge
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                
                old_sign = current[i]
                current[i] = -old_sign
                
                if is_lift_connected_fast(current):
                    new_radius = spectral_radius(current)
                    if new_radius < current_radius - 1e-10:
                        current_radius = new_radius
                        improved = True
                    else:
                        current[i] = old_sign
                else:
                    current[i] = old_sign
            
            if not improved:
                break
        
        return current, current_radius
    
    # Multiple random restarts
    rng = np.random.default_rng(3)
    best_signs = None
    best_radius = float('inf')
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initial signs
        signs = rng.choice([-1, 1], size=num_edges)
        
        # Ensure connectivity
        if not is_lift_connected_fast(signs):
            # Flip a random edge to try to fix
            for _ in range(100):
                idx = rng.integers(0, num_edges)
                signs[idx] = -signs[idx]
                if is_lift_connected_fast(signs):
                    break
        
        if not is_lift_connected_fast(signs):
            continue
        
        radius = spectral_radius(signs)
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
        
        # Local search
        signs_ls, radius_ls = local_search(signs)
        if radius_ls < best_radius:
            best_radius = radius_ls
            best_signs = signs_ls.copy()
    
    # If we still don't have a valid result, use a known valid pattern
    if best_signs is None:
        # All +1 except one edge set to -1
        best_signs = np.ones(num_edges, dtype=int)
        best_signs[0] = -1
    
    return best_signs.astype(np.int8)
