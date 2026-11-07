"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _build_lift_matrix(graph, signs):
    """Build the signed adjacency matrix for the base graph with given signs."""
    n = graph['n']
    edges = graph['edges']
    # The signed adjacency matrix A has A[u,v] = sign * 1 if (u,v) is an edge
    # For spectral radius computation, we need the matrix where each edge contributes
    # its sign. The spectral radius of this matrix gives the eigenvalues of the 2-lift.
    A = np.zeros((n, n))
    for i, (u, v) in enumerate(edges):
        A[u, v] = signs[i]
        A[v, u] = signs[i]
    return A

def _spectral_radius(A):
    """Compute spectral radius (max absolute eigenvalue) of symmetric matrix A."""
    try:
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    except Exception:
        return np.inf

def _is_connected_lift(graph, signs):
    """Check if the 2-lift defined by signs is connected."""
    n = graph['n']
    edges = graph['edges']
    
    # Build the lift graph with 2n vertices
    # Vertex (v, s) for v in 0..n-1, s in {0, 1}
    # For edge (u, v) with sign s:
    #   if s == +1: (u,0)-(v,0) and (u,1)-(v,1)
    #   if s == -1: (u,0)-(v,1) and (u,1)-(v,0)
    
    # Use BFS/DFS to check connectivity
    adj = [[] for _ in range(2 * n)]
    for i, (u, v) in enumerate(edges):
        s = signs[i]
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
    visited = [False] * (2 * n)
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        node = queue.pop()
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                count += 1
                queue.append(nb)
    
    return count == 2 * n

def _local_search(graph, signs, max_iterations=10000, time_limit=45.0):
    """Local search to minimize spectral radius while maintaining connectivity."""
    n = graph['n']
    edges = graph['edges']
    ne = len(edges)
    
    best_signs = signs.copy()
    best_A = _build_lift_matrix(graph, best_signs)
    best_radius = _spectral_radius(best_A)
    
    start_time = time.time()
    no_improve = 0
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if no_improve > 500:
            break
        
        # Try flipping each edge
        best_flip = -1
        best_new_radius = best_radius
        
        for i in range(ne):
            old_sign = best_signs[i]
            best_signs[i] = -old_sign
            
            A = _build_lift_matrix(graph, best_signs)
            radius = _spectral_radius(A)
            
            if radius < best_new_radius - 1e-12:
                best_new_radius = radius
                best_flip = i
            
            best_signs[i] = old_sign
        
        if best_flip >= 0:
            best_signs[best_flip] = -best_signs[best_flip]
            # Check connectivity
            if _is_connected_lift(graph, best_signs):
                best_A = _build_lift_matrix(graph, best_signs)
                best_radius = _spectral_radius(best_A)
                no_improve = 0
            else:
                # Revert if disconnected
                best_signs[best_flip] = -best_signs[best_flip]
                no_improve += 1
        else:
            no_improve += 1
    
    return best_signs, best_radius

def _random_search(graph, budget, time_limit=45.0):
    """Random search with local refinement."""
    n = graph['n']
    edges = graph['edges']
    ne = len(edges)
    
    best_signs = None
    best_radius = np.inf
    
    start_time = time.time()
    rng = np.random.default_rng(3)
    
    for sample in range(budget):
        if time.time() - start_time > time_limit:
            break
        
        signs = rng.choice([-1, 1], size=ne)
        
        if not _is_connected_lift(graph, signs):
            continue
        
        A = _build_lift_matrix(graph, signs)
        radius = _spectral_radius(A)
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all +1 signs (should be connected for prism graphs)
        best_signs = np.ones(ne, dtype=int)
        best_radius = _spectral_radius(_build_lift_matrix(graph, best_signs))
    
    return best_signs, best_radius

def sign_edges(graph, budget):
    """Return one sign (-1 or +1) per edge to minimize spectral radius of the 2-lift."""
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    n = graph['n']
    edges = graph['edges']
    ne = len(edges)
    
    # Phase 1: Random search to get an initial valid solution
    best_signs, best_radius = _random_search(graph, budget, time_limit=10.0)
    
    # Phase 2: Local search refinement
    if time.time() - start_time < deadline - 5.0:
        best_signs, best_radius = _local_search(graph, best_signs, max_iterations=20000, time_limit=deadline - 10.0)
    
    # Phase 3: Try more random restarts with local search if time permits
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 5.0:
        rng = np.random.default_rng(42)
        for restart in range(20):
            if time.time() - start_time > deadline - 3.0:
                break
            
            signs = rng.choice([-1, 1], size=ne)
            if not _is_connected_lift(graph, signs):
                continue
            
            # Quick local search from this starting point
            refined_signs, refined_radius = _local_search(graph, signs, max_iterations=2000, time_limit=2.0)
            
            if refined_radius < best_radius - 1e-12:
                best_radius = refined_radius
                best_signs = refined_signs
    
    # Final check: ensure validity
    if not _is_connected_lift(graph, best_signs):
        # Fallback to all +1
        best_signs = np.ones(ne, dtype=int)
    
    # Ensure correct shape and values
    best_signs = np.asarray(best_signs, dtype=int)
    best_signs = np.where(best_signs == 0, 1, best_signs)  # Safety
    
    return best_signs
# EVOLVE-BLOCK-END
