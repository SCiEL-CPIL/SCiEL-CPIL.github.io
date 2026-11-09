"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_signed_adjacency(graph, signs):
    """Build the signed adjacency matrix for the base graph."""
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n))
    for idx, (u, v) in enumerate(edges):
        A[u, v] = signs[idx]
        A[v, u] = signs[idx]
    return A

def _spectral_radius(graph, signs):
    """Compute the spectral radius of the signed adjacency matrix."""
    A = _build_signed_adjacency(graph, signs)
    # Use eigvalsh for symmetric matrix - returns eigenvalues in ascending order
    # Spectral radius is max absolute eigenvalue
    eigvals = np.linalg.eigvalsh(A)
    return np.max(np.abs(eigvals))

def _is_valid_lift(graph, signs):
    """Check if the lift is connected and 3-regular."""
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    
    # Build the lift graph
    # Vertices in lift: (v, 0) and (v, 1) for v in 0..n-1
    # Total 2*n vertices in lift
    num_lift_vertices = 2 * n
    adj = [[] for _ in range(num_lift_vertices)]
    
    for idx, (u, v) in enumerate(edges):
        s = signs[idx]
        if s == 1:
            # +1: joins equal sides
            # (u, 0) -- (v, 0) and (u, 1) -- (v, 1)
            adj[u * 2 + 0].append(v * 2 + 0)
            adj[v * 2 + 0].append(u * 2 + 0)
            adj[u * 2 + 1].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2 + 1)
        else:
            # -1: joins opposite sides
            # (u, 0) -- (v, 1) and (u, 1) -- (v, 0)
            adj[u * 2 + 0].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2 + 0)
            adj[u * 2 + 1].append(v * 2 + 0)
            adj[v * 2 + 0].append(u * 2 + 1)
    
    # Check 3-regular: each vertex should have exactly 3 neighbors
    for i in range(num_lift_vertices):
        if len(adj[i]) != 3:
            return False
    
    # Check connected: BFS/DFS from vertex 0
    visited = [False] * num_lift_vertices
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        node = stack.pop()
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                stack.append(neighbor)
    
    return count == num_lift_vertices

def sign_edges(graph, budget):
    """Return sign assignment for edges of a prism graph."""
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Generate initial candidate using the seed approach
    rng = np.random.default_rng(3)
    
    # Try random sign vectors and keep the best valid one
    best_signs = None
    best_radius = float('inf')
    
    # First, try the seed's budget random samples
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=num_edges)
        if _is_valid_lift(graph, signs):
            radius = _spectral_radius(graph, signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all +1 signs (should always be valid for prism graphs)
        best_signs = np.ones(num_edges, dtype=int)
        best_radius = _spectral_radius(graph, best_signs)
    
    # Local search: try flipping pairs of signs to escape local minima
    # Use 2-opt style moves
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try single flips
        for i in range(num_edges):
            if time.time() > deadline:
                break
            new_signs = best_signs.copy()
            new_signs[i] = -new_signs[i]
            if _is_valid_lift(graph, new_signs):
                radius = _spectral_radius(graph, new_signs)
                if radius < best_radius - 1e-12:
                    best_radius = radius
                    best_signs = new_signs
                    improved = True
        # Try 2-opt flips
        for i in range(num_edges):
            if time.time() > deadline:
                break
            for j in range(i + 1, num_edges):
                if time.time() > deadline:
                    break
                new_signs = best_signs.copy()
                new_signs[i] = -new_signs[i]
                new_signs[j] = -new_signs[j]
                if _is_valid_lift(graph, new_signs):
                    radius = _spectral_radius(graph, new_signs)
                    if radius < best_radius - 1e-12:
                        best_radius = radius
                        best_signs = new_signs
                        improved = True
    
    # Additional refinement: try more systematic patterns if time allows
    if time.time() < deadline - 5.0:
        # Try structured patterns based on cycle parity
        for pattern_type in range(4):
            if time.time() > deadline:
                break
            test_signs = np.ones(num_edges, dtype=int)
            for idx, (u, v) in enumerate(edges):
                if u < m and v < m:
                    # Bottom cycle edge
                    if pattern_type == 0:
                        test_signs[idx] = 1 if (u % 2 == 0) else -1
                    elif pattern_type == 1:
                        test_signs[idx] = -1 if (u % 2 == 0) else 1
                    elif pattern_type == 2:
                        test_signs[idx] = 1
                    else:
                        test_signs[idx] = -1
                elif u >= m and v >= m:
                    # Top cycle edge
                    if pattern_type == 0:
                        test_signs[idx] = -1 if ((u - m) % 2 == 0) else 1
                    elif pattern_type == 1:
                        test_signs[idx] = 1 if ((u - m) % 2 == 0) else -1
                    elif pattern_type == 2:
                        test_signs[idx] = 1
                    else:
                        test_signs[idx] = -1
                else:
                    # Matching edge
                    if pattern_type == 0:
                        test_signs[idx] = 1
                    elif pattern_type == 1:
                        test_signs[idx] = -1
                    elif pattern_type == 2:
                        test_signs[idx] = 1
                    else:
                        test_signs[idx] = -1
            
            if _is_valid_lift(graph, test_signs):
                radius = _spectral_radius(graph, test_signs)
                if radius < best_radius - 1e-12:
                    best_radius = radius
                    best_signs = test_signs.copy()
                    # Continue local search from this new starting point
                    improved2 = True
                    while improved2 and time.time() < deadline:
                        improved2 = False
                        for i in range(num_edges):
                            if time.time() > deadline:
                                break
                            new_signs = best_signs.copy()
                            new_signs[i] = -new_signs[i]
                            if _is_valid_lift(graph, new_signs):
                                radius = _spectral_radius(graph, new_signs)
                                if radius < best_radius - 1e-12:
                                    best_radius = radius
                                    best_signs = new_signs
                                    improved2 = True
    
    return best_signs
# EVOLVE-BLOCK-END
