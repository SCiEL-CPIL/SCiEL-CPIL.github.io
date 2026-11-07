"""Optimized solver for spectral graph lifts on prism graphs."""
import time
import numpy as np
from itertools import product

def _build_signed_adjacency(graph, signs):
    """Build the signed adjacency matrix from graph and signs."""
    n = graph['n']
    A = np.zeros((n, n), dtype=float)
    for idx, (u, v) in enumerate(graph['edges']):
        s = float(signs[idx])
        A[u, v] = s
        A[v, u] = s
    return A

def _spectral_radius(A):
    """Compute spectral radius (max absolute eigenvalue) of symmetric A."""
    try:
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    except:
        return float('inf')

def _is_connected_lift(graph, signs):
    """Check if the 2-lift defined by signs is connected."""
    n = graph['n']
    # Lift has 2n vertices: (v, 0) and (v, 1)
    # Edge (u, v) with sign s:
    #   if s == +1: (u,0)-(v,0) and (u,1)-(v,1)
    #   if s == -1: (u,0)-(v,1) and (u,1)-(v,0)
    
    # Build adjacency for the lift
    lift_edges = []
    for idx, (u, v) in enumerate(graph['edges']):
        s = signs[idx]
        if s == 1:
            lift_edges.append((u * 2, v * 2))
            lift_edges.append((u * 2 + 1, v * 2 + 1))
        else:
            lift_edges.append((u * 2, v * 2 + 1))
            lift_edges.append((u * 2 + 1, v * 2))
    
    # BFS/DFS to check connectivity
    num_lift_vertices = 2 * n
    visited = [False] * num_lift_vertices
    adj = [[] for _ in range(num_lift_vertices)]
    for a, b in lift_edges:
        adj[a].append(b)
        adj[b].append(a)
    
    # Start from vertex 0
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

def _evaluate(graph, signs):
    """Evaluate a sign assignment: return (spectral_radius, is_valid)."""
    if not _is_connected_lift(graph, signs):
        return float('inf'), False
    A = _build_signed_adjacency(graph, signs)
    radius = _spectral_radius(A)
    return radius, True

def _local_search(graph, signs, max_iters=2000, timeout_sec=40):
    """Hill-climbing local search to minimize spectral radius."""
    n_edges = len(graph['edges'])
    signs = signs.copy()
    best_radius, _ = _evaluate(graph, signs)
    
    start_time = time.time()
    improved = True
    iters = 0
    
    while improved and iters < max_iters and (time.time() - start_time) < timeout_sec:
        improved = False
        iters += 1
        # Try flipping each edge
        for i in range(n_edges):
            signs[i] = -signs[i]
            radius, valid = _evaluate(graph, signs)
            if valid and radius < best_radius - 1e-12:
                best_radius = radius
                improved = True
            else:
                signs[i] = -signs[i]  # revert
    
    return signs, best_radius

def sign_edges(graph, budget):
    """Return sign assignment minimizing spectral radius of the lift."""
    start_time = time.time()
    deadline = 50.0  # safety margin before 60s limit
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    n_edges = len(edges)
    
    # Identify edge types:
    # Cycle edges on top: (i, i+1) for i in 0..m-1 (with wrap)
    # Cycle edges on bottom: (m+i, m+i+1) for i in 0..m-1 (with wrap)
    # Matching edges: (i, m+i) for i in 0..m-1
    
    # For prism graph, edges are typically ordered as:
    # top cycle, bottom cycle, matching
    # But let's be robust: classify by vertex ranges
    
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(idx)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(idx)
        else:
            matching_edges.append(idx)
    
    # Sort each group for consistency
    top_cycle_edges.sort()
    bottom_cycle_edges.sort()
    matching_edges.sort()
    
    # Strategy: exploit structure
    # The prism graph has a natural symmetry. Let's try structured sign patterns.
    
    best_signs = None
    best_radius = float('inf')
    
    def try_signs(signs):
        nonlocal best_signs, best_radius
        radius, valid = _evaluate(graph, signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # Phase 1: Try structured patterns based on the prism symmetry
    # For a prism, a "balanced" sign pattern might work well.
    # Try all combinations where top and bottom cycles have consistent signs
    # and matching has some pattern.
    
    # First, try a few heuristic patterns
    # Pattern 1: all +1
    signs = np.ones(n_edges, dtype=int)
    try_signs(signs)
    
    # Pattern 2: alternating on cycles, +1 on matching
    signs = np.ones(n_edges, dtype=int)
    for idx in top_cycle_edges:
        pos = top_cycle_edges.index(idx)
        signs[idx] = 1 if pos % 2 == 0 else -1
    for idx in bottom_cycle_edges:
        pos = bottom_cycle_edges.index(idx)
        signs[idx] = 1 if pos % 2 == 0 else -1
    try_signs(signs)
    
    # Phase 2: Systematic search over small parameter space
    # For the prism, we can parameterize by:
    # - sign of each matching edge (m bits)
    # - relative sign pattern of top vs bottom cycle
    
    # Let's do a more thorough local search from multiple starting points
    
    # Starting point 1: all +1
    signs_start = np.ones(n_edges, dtype=int)
    signs_opt, rad = _local_search(graph, signs_start, max_iters=1000, timeout_sec=10)
    try_signs(signs_opt)
    
    if time.time() - start_time > deadline:
        return best_signs if best_signs is not None else np.ones(n_edges, dtype=int)
    
    # Starting point 2: alternating cycles
    signs_start = np.ones(n_edges, dtype=int)
    for idx in top_cycle_edges:
        pos = top_cycle_edges.index(idx)
        signs_start[idx] = 1 if pos % 2 == 0 else -1
    for idx in bottom_cycle_edges:
        pos = bottom_cycle_edges.index(idx)
        signs_start[idx] = 1 if pos % 2 == 0 else -1
    signs_opt, rad = _local_search(graph, signs_start, max_iters=1000, timeout_sec=10)
    try_signs(signs_opt)
    
    if time.time() - start_time > deadline:
        return best_signs if best_signs is not None else np.ones(n_edges, dtype=int)
    
    # Phase 3: Random restarts with local search
    rng = np.random.default_rng(3)
    
    for restart in range(100):
        if time.time() - start_time > deadline - 5:
            break
        
        # Random sign assignment
        signs_rand = rng.choice([-1, 1], size=n_edges)
        signs_opt, rad = _local_search(graph, signs_rand, max_iters=500, timeout_sec=5)
        try_signs(signs_opt)
    
    if time.time() - start_time > deadline:
        return best_signs if best_signs is not None else np.ones(n_edges, dtype=int)
    
    # Phase 4: More thorough local search from the best found
    if best_signs is not None:
        signs_opt, rad = _local_search(graph, best_signs, max_iters=5000, timeout_sec=20)
        try_signs(signs_opt)
    
    return best_signs if best_signs is not None else np.ones(n_edges, dtype=int)
