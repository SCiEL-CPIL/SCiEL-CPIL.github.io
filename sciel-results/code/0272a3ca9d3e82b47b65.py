"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_signed_matrix(n, edges, signs):
    """Build the signed adjacency matrix."""
    S = np.zeros((n, n), dtype=np.float64)
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        S[u, v] = s
        S[v, u] = s
    return S

def _spectral_radius(S):
    """Compute spectral radius (max absolute eigenvalue) of symmetric matrix S."""
    vals = np.linalg.eigvalsh(S)
    return np.max(np.abs(vals))

def _is_connected_lift(n, edges, signs):
    """Check if the 2-lift is connected.
    
    The lift has 2n vertices: (v, 0) and (v, 1) for v in 0..n-1.
    Edge (u,v) with sign +1 connects (u,0)-(v,0) and (u,1)-(v,1).
    Edge (u,v) with sign -1 connects (u,0)-(v,1) and (u,1)-(v,0).
    """
    # BFS/DFS on the lift graph
    visited = [False] * (2 * n)
    # Start from vertex (0, 0)
    stack = [0]  # vertex index in lift: v*2 + side
    visited[0] = True
    count = 1
    
    # Build adjacency list for the lift
    lift_adj = [[] for _ in range(2 * n)]
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            # +1: equal sides
            lift_adj[u * 2].append(v * 2)
            lift_adj[v * 2].append(u * 2)
            lift_adj[u * 2 + 1].append(v * 2 + 1)
            lift_adj[v * 2 + 1].append(u * 2 + 1)
        else:
            # -1: opposite sides
            lift_adj[u * 2].append(v * 2 + 1)
            lift_adj[v * 2 + 1].append(u * 2)
            lift_adj[u * 2 + 1].append(v * 2)
            lift_adj[v * 2].append(u * 2 + 1)
    
    while stack:
        node = stack.pop()
        for neighbor in lift_adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                stack.append(neighbor)
    
    return count == 2 * n

def _evaluate(n, edges, signs):
    """Return spectral radius if valid, else infinity."""
    if not _is_connected_lift(n, edges, signs):
        return np.inf
    S = _build_signed_matrix(n, edges, signs)
    return _spectral_radius(S)

def _local_search(n, edges, signs, deadline, max_iter=5000):
    """Local search with 1-flip and 2-flip moves."""
    current = signs.copy()
    current_radius = _evaluate(n, edges, current)
    if np.isinf(current_radius):
        return current, current_radius
    
    best = current.copy()
    best_radius = current_radius
    num_edges = len(edges)
    
    # Precompute edge indices for quick access
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
    
    iter_count = 0
    while time.time() < deadline - 1.0 and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        # Try 1-flip moves
        for i in range(num_edges):
            if time.time() > deadline - 1.0:
                break
            trial = current.copy()
            trial[i] = -trial[i]
            radius = _evaluate(n, edges, trial)
            if radius < current_radius - 1e-12:
                current = trial
                current_radius = radius
                improved = True
                if radius < best_radius:
                    best = current.copy()
                    best_radius = radius
                break  # Restart the flip search after improvement
        
        if not improved:
            # Try 2-flip moves
            two_flip_improved = False
            # Sample pairs to avoid O(n^2)
            num_pairs_to_try = min(200, num_edges * (num_edges - 1) // 2)
            for _ in range(num_pairs_to_try):
                if time.time() > deadline - 1.0:
                    break
                i, j = np.random.default_rng(iter_count).choice(num_edges, size=2, replace=False)
                if i == j:
                    continue
                trial = current.copy()
                trial[i] = -trial[i]
                trial[j] = -trial[j]
                radius = _evaluate(n, edges, trial)
                if radius < current_radius - 1e-12:
                    current = trial
                    current_radius = radius
                    two_flip_improved = True
                    if radius < best_radius:
                        best = current.copy()
                        best_radius = radius
                    break
            
            if not two_flip_improved:
                break
    
    return best, best_radius

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy: Random restarts + local search to minimize spectral radius.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    best_signs = None
    best_radius = np.inf
    
    # First, get a valid starting point using the seed's approach
    from autoresearch_suite.science_tcs_expansion import lift_baseline
    try:
        seed_signs = lift_baseline(graph, budget)
        seed_radius = _evaluate(n, edges, seed_signs)
        if np.isinf(seed_radius):
            seed_radius = np.inf
        best_signs = seed_signs.copy()
        best_radius = seed_radius
    except Exception:
        pass
    
    # If we didn't get a valid seed, try random until we find one
    if best_signs is None or np.isinf(best_radius):
        rng = np.random.default_rng(3)
        for _ in range(1000):
            if time.time() > deadline - 5.0:
                break
            signs = rng.choice([-1, 1], size=num_edges)
            radius = _evaluate(n, edges, signs)
            if not np.isinf(radius):
                best_signs = signs
                best_radius = radius
                break
    
    if best_signs is None:
        # Emergency: return all +1 (may not be valid, but better than exception)
        return np.ones(num_edges, dtype=np.int8)
    
    # Local search with restarts
    rng = np.random.default_rng(42)
    restart = 0
    while time.time() < deadline - 2.0:
        # Run local search from current best
        improved_signs, improved_radius = _local_search(n, edges, best_signs, deadline, max_iter=3000)
        
        if improved_radius < best_radius - 1e-12:
            best_signs = improved_signs
            best_radius = improved_radius
        
        # Random restart: perturb the best solution
        if time.time() > deadline - 2.0:
            break
        
        # Generate a new starting point by random perturbation of best
        num_flips = rng.integers(1, min(5, num_edges))
        flip_indices = rng.choice(num_edges, size=num_flips, replace=False)
        new_start = best_signs.copy()
        new_start[flip_indices] = -new_start[flip_indices]
        
        # Check if new start is valid
        new_radius = _evaluate(n, edges, new_start)
        if not np.isinf(new_radius):
            # Run local search from this new start
            ls_signs, ls_radius = _local_search(n, edges, new_start, deadline, max_iter=2000)
            if ls_radius < best_radius - 1e-12:
                best_signs = ls_signs
                best_radius = ls_radius
        
        restart += 1
    
    return best_signs.astype(np.int8)
# EVOLVE-BLOCK-END
