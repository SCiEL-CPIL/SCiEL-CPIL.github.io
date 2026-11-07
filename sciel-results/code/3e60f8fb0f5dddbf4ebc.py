import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m
    
    # Build adjacency structure
    adj = {i: [] for i in range(n_vertices)}
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    def build_signed_adj(signs):
        A = np.zeros((n_vertices, n_vertices), dtype=float)
        for idx, (u, v) in enumerate(edges):
            s = float(signs[idx])
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    
    def is_connected_lift(signs):
        # Check if the 2-lift is connected
        # Build the lift graph and check connectivity
        # Lift has 2*n_vertices vertices
        # Edge (u,v) with sign +1: (u,0)-(v,0), (u,1)-(v,1)
        # Edge (u,v) with sign -1: (u,0)-(v,1), (u,1)-(v,0)
        
        lift_adj = [[] for _ in range(2 * n_vertices)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                # Connect same sides
                lift_adj[u * 2].append(v * 2)
                lift_adj[v * 2].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2 + 1)
            else:
                # Connect opposite sides
                lift_adj[u * 2].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2)
                lift_adj[v * 2].append(u * 2 + 1)
        
        # BFS/DFS to check connectivity
        visited = [False] * (2 * n_vertices)
        queue = [0]
        visited[0] = True
        while queue:
            node = queue.pop()
            for neighbor in lift_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        return all(visited)
    
    def is_valid(signs):
        return is_connected_lift(signs)
    
    # Initial assignment: try to make it valid
    # Start with all +1, which gives a disconnected lift (two copies of the graph)
    # We need to introduce some -1 signs to make it connected
    
    # Strategy: start with a pattern that's likely valid
    # For prism graphs, alternating signs on the matching might work
    
    best_signs = None
    best_radius = float('inf')
    
    # Try several initial patterns
    rng = np.random.default_rng(42)
    
    # Pattern 1: all +1 except one matching edge is -1
    for init_pattern in range(5):
        if time.time() > deadline:
            break
        
        signs = np.ones(n_edges, dtype=int)
        
        if init_pattern == 0:
            # All +1
            pass
        elif init_pattern == 1:
            # One -1 on matching
            # Find matching edges (those connecting i to i+m)
            for idx, (u, v) in enumerate(edges):
                if abs(u - v) == m:
                    signs[idx] = -1
                    break
        elif init_pattern == 2:
            # Alternating on top cycle
            for idx, (u, v) in enumerate(edges):
                if u < m and v < m:  # top cycle edge
                    if (u + v) % 2 == 0:
                        signs[idx] = -1
        elif init_pattern == 3:
            # Random
            signs = rng.choice([-1, 1], size=n_edges)
        elif init_pattern == 4:
            # All -1 on matching
            for idx, (u, v) in enumerate(edges):
                if abs(u - v) == m:
                    signs[idx] = -1
        
        if is_valid(signs):
            r = spectral_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    
    # If no valid initial found, do random search for a valid one
    if best_signs is None:
        for _ in range(1000):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=n_edges)
            if is_valid(signs):
                best_signs = signs.copy()
                best_radius = spectral_radius(signs)
                break
    
    if best_signs is None:
        # Fallback: return all +1 (may be invalid but better than nothing)
        return np.ones(n_edges, dtype=int)
    
    # Local search: flip individual edges
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # Precompute which edges are which type for faster evaluation
    edge_types = []
    for u, v in edges:
        if u < m and v < m:
            edge_types.append('top')
        elif u >= m and v >= m:
            edge_types.append('bottom')
        else:
            edge_types.append('match')
    
    # Local search with single edge flips
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n_edges):
            if time.time() > deadline:
                break
            
            # Try flipping edge idx
            new_signs = current_signs.copy()
            new_signs[idx] = -new_signs[idx]
            
            if is_valid(new_signs):
                new_radius = spectral_radius(new_signs)
                if new_radius < current_radius - 1e-10:
                    current_signs = new_signs
                    current_radius = new_radius
                    improved = True
    
    # Try double edge flips for more exploration
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            # Pick two random edges to flip
            idx1 = rng.integers(0, n_edges)
            idx2 = rng.integers(0, n_edges)
            if idx1 == idx2:
                continue
            
            new_signs = current_signs.copy()
            new_signs[idx1] = -new_signs[idx1]
            new_signs[idx2] = -new_signs[idx2]
            
            if is_valid(new_signs):
                new_radius = spectral_radius(new_signs)
                if new_radius < current_radius - 1e-10:
                    current_signs = new_signs
                    current_radius = new_radius
    
    # Final check
    if is_valid(current_signs):
        return current_signs
    else:
        # Return best valid found
        return best_signs
