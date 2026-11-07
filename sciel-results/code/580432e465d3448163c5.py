def sign_edges(graph, budget):
    import numpy as np
    import time
    
    start_time = time.time()
    DEADLINE = 55  # seconds, safety margin before 60s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)
    
    # Build adjacency structure for the base graph
    # For each vertex, list of (neighbor, edge_index)
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Classify edges: 0=bottom cycle, 1=top cycle, 2=matching
    edge_type = np.zeros(E, dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            edge_type[i] = 0  # bottom
        elif u >= m and v >= m:
            edge_type[i] = 1  # top
        else:
            edge_type[i] = 2  # matching
    
    # Build signed adjacency matrix from signs
    def build_signed_adj(signs):
        A = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            A[u, v] += s
            A[v, u] += s
        return A
    
    def spectral_radius(signs):
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    
    def is_connected_lift(signs):
        # Check if the 2-lift is connected
        # Vertices in lift: (v, 0) and (v, 1) for v in 0..n-1
        # Total 2n vertices in lift
        # BFS/DFS on the lift
        visited = np.zeros(2 * n, dtype=bool)
        queue = [0]  # start from (0, 0)
        visited[0] = True
        count = 1
        
        while queue:
            node = queue.pop(0)
            v = node // 2
            side = node % 2
            for (neighbor, ei) in adj[v]:
                s = signs[ei]
                if s == 1:
                    new_side = side
                else:
                    new_side = 1 - side
                new_node = neighbor * 2 + new_side
                if not visited[new_node]:
                    visited[new_node] = True
                    count += 1
                    queue.append(new_node)
        
        return count == 2 * n
    
    # Initial sign assignment: try to get a good starting point
    # The seed uses random sampling with budget=32
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = float('inf')
    
    # Try the deterministic assignment first
    det_signs = np.zeros(E, dtype=np.int8)
    for i in range(E):
        if edge_type[i] == 0:
            det_signs[i] = 1
        elif edge_type[i] == 1:
            det_signs[i] = -1
        else:
            det_signs[i] = 1
    
    if is_connected_lift(det_signs):
        r = spectral_radius(det_signs)
        if r < best_radius:
            best_radius = r
            best_signs = det_signs.copy()
    
    # Try random initializations
    for trial in range(min(budget, 32)):
        signs = rng.choice([-1, 1], size=E).astype(np.int8)
        if is_connected_lift(signs):
            r = spectral_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: use deterministic
        best_signs = det_signs.copy()
        best_radius = spectral_radius(best_signs)
    
    # Local search: flip one edge at a time
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # Precompute which edges are in which cycles for connectivity checking
    # Actually, just check connectivity each time - it's O(n) which is fine for small graphs
    
    flip_order = list(range(E))
    
    # Simulated annealing style local search
    temperature = 0.1
    cooling = 0.99
    
    while time.time() - start_time < DEADLINE:
        np.random.shuffle(flip_order)
        improved = False
        
        for ei in flip_order:
            if time.time() - start_time > DEADLINE:
                break
            
            # Try flipping edge ei
            trial_signs = current_signs.copy()
            trial_signs[ei] *= -1
            
            # Check connectivity
            if not is_connected_lift(trial_signs):
                continue
            
            # Compute spectral radius
            trial_radius = spectral_radius(trial_signs)
            
            # Accept if better, or with some probability if worse (simulated annealing)
            if trial_radius < current_radius:
                current_signs = trial_signs
                current_radius = trial_radius
                improved = True
            else:
                # Accept with probability exp(-delta/T)
                delta = trial_radius - current_radius
                if delta < 0.01 and np.random.random() < np.exp(-delta / max(temperature, 1e-6)):
                    current_signs = trial_signs
                    current_radius = trial_radius
                    improved = True
        
        temperature *= cooling
        if temperature < 1e-4:
            temperature = 0.01  # Reset temperature occasionally
    
    # Final check
    if not is_connected_lift(current_signs):
        current_signs = best_signs
    else:
        final_r = spectral_radius(current_signs)
        if final_r < best_radius:
            best_signs = current_signs
            best_radius = final_r
    
    return best_signs.astype(np.int8)
