"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _build_base_adjacency(graph):
    n = graph['n']
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v) in graph['edges']:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A

def _signed_radius(A, signs):
    n = A.shape[0]
    S = np.diag(signs)
    B = A @ S
    eigvals = np.linalg.eigvalsh(B)
    return np.max(np.abs(eigvals))

def _is_connected_lift(graph, signs):
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    # Build lift graph: 2*n vertices, vertex (v, side) -> v*2 + side
    adj = [[] for _ in range(2 * n)]
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        # Edge (u,v) with sign s: connects (u,0) to (v,0) if s=+1, else (u,0) to (v,1)
        # and (u,1) to (v,1) if s=+1, else (u,1) to (v,0)
        if s == 1:
            adj[u*2].append(v*2)
            adj[v*2].append(u*2)
            adj[u*2+1].append(v*2+1)
            adj[v*2+1].append(u*2+1)
        else:
            adj[u*2].append(v*2+1)
            adj[v*2+1].append(u*2)
            adj[u*2+1].append(v*2)
            adj[v*2].append(u*2+1)
    
    # BFS from (0,0)
    visited = [False] * (2 * n)
    visited[0] = True
    stack = [0]
    count = 1
    while stack:
        node = stack.pop()
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                stack.append(neighbor)
    return count == 2 * n

def _is_valid_lift(graph, signs):
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    if len(signs) != num_edges:
        return False
    # Check that lift is 3-regular
    # Each vertex in base has degree 3 (for prism graphs), so lift is automatically 3-regular
    # if the lift is connected. Let's just check connectivity.
    return _is_connected_lift(graph, signs)

def _try_signs_for_m(m, num_edges, edge_list, time_limit=55.0):
    start_time = time.time()
    best_radius = np.inf
    best_signs = None
    
    # For m=4: 12 edges, 2^12 = 4096 possibilities
    # For m=5: 15 edges, 2^15 = 32768
    # For m=7: 21 edges, 2^21 = 2M - too many for brute force
    # For m=9: 27 edges, 2^27 = 134M - way too many
    
    # Strategy: 
    # 1. Try all sign assignments for small m (4, 5)
    # 2. For larger m, use a smarter approach
    
    # First, identify which edges are cycle edges vs matching edges
    # Prism graph structure:
    # Vertices 0..m-1 form a cycle, vertices m..2m-1 form a cycle
    # Matching edges: (i, i+m) for i in 0..m-1
    # Cycle edges: (i, (i+1)%m) for i in 0..m-1, and (m+i, m+(i+1)%m) for i in 0..m-1
    
    # Let's identify edge types
    cycle_edges = []
    matching_edges = []
    for i, (u, v) in enumerate(edge_list):
        if abs(u - v) == m:
            matching_edges.append(i)
        else:
            cycle_edges.append(i)
    
    # For the lift to be connected, we need certain conditions on the signs.
    # The key insight: for a prism graph, the lift is connected iff the product of 
    # signs around each "face" satisfies certain conditions, or more simply, 
    # we can enumerate smartly.
    
    # Actually, let's think about this differently. The signed adjacency matrix 
    # eigenvalues are what we want to minimize the max absolute value of.
    
    # For small m, brute force with early termination
    if num_edges <= 15:
        # Brute force all 2^num_edges sign assignments
        for mask in range(1 << num_edges):
            if time.time() - start_time > time_limit:
                break
            signs = np.zeros(num_edges, dtype=np.float64)
            for i in range(num_edges):
                signs[i] = 1.0 if (mask >> i) & 1 else -1.0
            
            if not _is_valid_lift(graph_helper(m), signs):
                continue
            
            A = _build_base_adjacency(graph_helper(m))
            radius = _signed_radius(A, signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        return best_signs, best_radius
    
    # For larger m, use a heuristic approach
    # Start with all +1, then flip edges to reduce spectral radius
    signs = np.ones(num_edges, dtype=np.float64)
    if not _is_valid_lift(graph_helper(m), signs):
        # Try all -1
        signs = -np.ones(num_edges, dtype=np.float64)
        if not _is_valid_lift(graph_helper(m), signs):
            # Try a pattern
            for i in range(num_edges):
                signs[i] = 1.0 if i % 2 == 0 else -1.0
            if not _is_valid_lift(graph_helper(m), signs):
                # Just use the first valid one we find via random search
                rng = np.random.default_rng(42)
                while time.time() - start_time < time_limit:
                    signs = rng.choice([-1.0, 1.0], size=num_edges)
                    if _is_valid_lift(graph_helper(m), signs):
                        break
    
    A = _build_base_adjacency(graph_helper(m))
    best_radius = _signed_radius(A, signs)
    best_signs = signs.copy()
    
    # Local search: flip one edge at a time
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(num_edges):
            signs[i] = -signs[i]
            if _is_valid_lift(graph_helper(m), signs):
                radius = _signed_radius(A, signs)
                if radius < best_radius - 1e-12:
                    best_radius = radius
                    best_signs = signs.copy()
                    improved = True
                    signs[i] = -signs[i]  # Keep the flip
                    break
            else:
                signs[i] = -signs[i]  # Revert
    
    # Two-edge flips
    if (time.time() - start_time) < time_limit:
        for i in range(num_edges):
            for j in range(i + 1, num_edges):
                if time.time() - start_time > time_limit:
                    break
                signs[i] = -signs[i]
                signs[j] = -signs[j]
                if _is_valid_lift(graph_helper(m), signs):
                    radius = _signed_radius(A, signs)
                    if radius < best_radius - 1e-12:
                        best_radius = radius
                        best_signs = signs.copy()
                        improved = True
                        # Keep the flips
                    else:
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]
                else:
                    signs[i] = -signs[i]
                    signs[j] = -signs[j]
                if improved:
                    break
            if improved:
                break
    
    return best_signs, best_radius

def graph_helper(m):
    n = 2 * m
    edges = []
    # First cycle: 0..m-1
    for i in range(m):
        edges.append((i, (i + 1) % m))
    # Second cycle: m..2m-1
    for i in range(m):
        edges.append((m + i, m + (i + 1) % m))
    # Matching
    for i in range(m):
        edges.append((i, i + m))
    return {'n': n, 'edges': edges}

def sign_edges(graph, budget):
    m = graph['n'] // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    start_time = time.time()
    time_limit = 55.0
    
    if num_edges <= 15:
        best_signs, best_radius = _try_signs_for_m(m, num_edges, edges, time_limit)
    else:
        best_signs, best_radius = _try_signs_for_m(m, num_edges, edges, time_limit)
    
    if best_signs is None:
        # Fallback: use random valid signs
        from autoresearch_suite.science_tcs_expansion import lift_baseline
        return lift_baseline(graph, budget)
    
    return best_signs.astype(np.float64)
# EVOLVE-BLOCK-END
