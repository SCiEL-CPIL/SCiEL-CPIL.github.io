def sign_edges(graph, budget):
    import numpy as np
    import time
    from collections import deque
    
    start_time = time.time()
    DEADLINE = 85  # Safety margin
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m
    
    # Precompute edge structure
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    
    # Precompute adjacency for lift (as neighbor lists for each base vertex)
    # For each base vertex, store list of (neighbor_base_vertex, edge_index)
    base_adj = [[] for _ in range(n_vertices)]
    for i, (u, v) in enumerate(edges):
        base_adj[u].append((v, i))
        base_adj[v].append((u, i))
    
    def build_signed_adj(signs):
        A = np.zeros((n_vertices, n_vertices))
        for i in range(n_edges):
            s = signs[i]
            A[edge_u[i], edge_v[i]] = s
            A[edge_v[i], edge_u[i]] = s
        return A
        
    def spectral_radius(signs):
        A = build_signed_adj(signs)
        return np.max(np.abs(np.linalg.eigvalsh(A)))
        
    def is_valid_lift(signs):
        """Check if the 2-lift is connected using BFS."""
        n_lift = 2 * n_vertices
        visited = np.zeros(n_lift, dtype=bool)
        visited[0] = True
        queue = deque([0])
        count = 1
        
        while queue:
            node = queue.popleft()
            base_node = node >> 1
            side = node & 1
            for nb, ei in base_adj[base_node]:
                s = signs[ei]
                # If sign is +1, stay on same side; if -1, switch side
                nb_side = side if s == 1 else (1 - side)
                nb_lift = (nb << 1) | nb_side
                if not visited[nb_lift]:
                    visited[nb_lift] = True
                    count += 1
                    queue.append(nb_lift)
                    if count == n_lift:
                        return True
        return count == n_lift

    def try_flip(signs, i):
        """Flip sign at index i, return new signs or None if invalid."""
        signs[i] = -signs[i]
        if not is_valid_lift(signs):
            signs[i] = -signs[i]
            return None
        return signs

    best_signs = None
    best_radius = float('inf')
    
    rng = np.random.default_rng(3)
    
    # Phase 1: Find a valid starting point
    valid_signs = None
    for attempt in range(200):
        if attempt == 0:
            # Start with all +1, then flip one matching edge to make it connected
            candidate = np.ones(n_edges, dtype=int)
            # Find a matching edge (i, i+m)
            for i, (u, v) in enumerate(edges):
                if abs(u - v) == m:
                    candidate[i] = -1
                    break
            if is_valid_lift(candidate):
                valid_signs = candidate
                break
        else:
            candidate = rng.choice([-1, 1], size=n_edges)
            if is_valid_lift(candidate):
                valid_signs = candidate
                break
    
    if valid_signs is None:
        # Fallback: random until valid
        valid_signs = rng.choice([-1, 1], size=n_edges)
        while not is_valid_lift(valid_signs):
            idx = rng.integers(0, n_edges)
            valid_signs[idx] *= -1
    
    best_signs = valid_signs.copy()
    best_radius = spectral_radius(best_signs)
    
    # Phase 2: Local search with multiple restarts
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Initialize current solution
        if restart == 0:
            current_signs = best_signs.copy()
        else:
            current_signs = best_signs.copy()
            # Perturb with 1-3 random flips
            n_flips = rng.integers(1, 4)
            for _ in range(n_flips):
                idx = rng.integers(0, n_edges)
                current_signs[idx] *= -1
            if not is_valid_lift(current_signs):
                continue
        
        current_radius = spectral_radius(current_signs)
        
        # Local search loop
        improved = True
        stagnation = 0
        while improved and (time.time() - start_time < DEADLINE - 2):
            improved = False
            stagnation = 0
            
            # Try all 1-flips
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE - 2:
                    break
                old_val = current_signs[i]
                current_signs[i] = -old_val
                
                if is_valid_lift(current_signs):
                    new_radius = spectral_radius(current_signs)
                    if new_radius < current_radius - 1e-10:
                        current_radius = new_radius
                        improved = True
                    else:
                        current_signs[i] = old_val
                else:
                    current_signs[i] = old_val
            
            # If stuck, try random 2-flips
            if not improved:
                stagnation += 1
                for _ in range(20):
                    if time.time() - start_time > DEADLINE - 2:
                        break
                    i, j = rng.choice(n_edges, size=2, replace=False)
                    oi, oj = current_signs[i], current_signs[j]
                    current_signs[i] = -oi
                    current_signs[j] = -oj
                    
                    if is_valid_lift(current_signs):
                        new_radius = spectral_radius(current_signs)
                        if new_radius < current_radius - 1e-10:
                            current_radius = new_radius
                            improved = True
                            break
                        else:
                            current_signs[i] = oi
                            current_signs[j] = oj
                    else:
                        current_signs[i] = oi
                        current_signs[j] = oj
                
                if not improved and stagnation > 3:
                    break
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()
    
    # Phase 3: Final polish - try all single flips on the best solution
    if time.time() - start_time < DEADLINE - 3:
        current_signs = best_signs.copy()
        current_radius = best_radius
        for i in range(n_edges):
            if time.time() - start_time > DEADLINE - 3:
                break
            old_val = current_signs[i]
            current_signs[i] = -old_val
            if is_valid_lift(current_signs):
                new_radius = spectral_radius(current_signs)
                if new_radius < current_radius - 1e-12:
                    current_radius = new_radius
                    best_signs = current_signs.copy()
                    best_radius = new_radius
                else:
                    current_signs[i] = old_val
            else:
                current_signs[i] = old_val
    
    return best_signs
