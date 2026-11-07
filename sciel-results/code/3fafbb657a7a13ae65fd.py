"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_adjacency(graph):
    n = graph['n']
    edges = graph['edges']
    return n, edges

def _signed_adj_matrix(n, edges, signs):
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v), s in zip(edges, signs):
        A[u, v] = s
        A[v, u] = s
    return A

def _spectral_radius(n, edges, signs):
    A = _signed_adj_matrix(n, edges, signs)
    evals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(evals)))

def _is_connected_lift(n, edges, signs):
    # Build the lift graph
    # Lift has 2n vertices: (v, 0) and (v, 1)
    # For each edge (u, v) with sign s:
    #   if s == +1: (u,0)-(v,0) and (u,1)-(v,1)
    #   if s == -1: (u,0)-(v,1) and (u,1)-(v,0)
    # Check connectivity of the lift graph
    
    num_lift_vertices = 2 * n
    # Adjacency list for lift
    adj = [[] for _ in range(num_lift_vertices)]
    
    for (u, v), s in zip(edges, signs):
        if s == 1:
            adj[u * 2].append(v * 2)
            adj[v * 2].append(u * 2)
            adj[u * 2 + 1].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2 + 1)
        else:
            adj[u * 2].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2)
            adj[u * 2 + 1].append(v * 2)
            adj[v * 2].append(u * 2 + 1)
    
    # BFS/DFS from vertex 0
    visited = [False] * num_lift_vertices
    queue = [0]
    visited[0] = True
    count = 1
    
    while queue:
        curr = queue.pop()
        for neighbor in adj[curr]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                queue.append(neighbor)
    
    return count == num_lift_vertices

def _evaluate(n, edges, signs):
    """Return (radius, valid) for a given sign assignment."""
    if not _is_connected_lift(n, edges, signs):
        return None, False
    radius = _spectral_radius(n, edges, signs)
    return radius, True

def _local_search(n, edges, signs, max_iter=5000, time_limit=50.0):
    """Perform local search to minimize spectral radius."""
    current_signs = list(signs)
    current_radius, valid = _evaluate(n, edges, current_signs)
    
    if not valid:
        return current_signs, None
    
    num_edges = len(edges)
    start_time = time.time()
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        best_radius = current_radius
        best_idx = -1
        
        # Try flipping each edge
        for i in range(num_edges):
            old_val = current_signs[i]
            current_signs[i] = -old_val
            
            radius, valid = _evaluate(n, edges, current_signs)
            
            if valid and radius < best_radius:
                best_radius = radius
                best_idx = i
            
            current_signs[i] = old_val
        
        if best_idx == -1:
            break
        
        current_signs[best_idx] = -current_signs[best_idx]
        current_radius = best_radius
    
    return current_signs, current_radius

def sign_edges(graph, budget):
    n, edges = _build_adjacency(graph)
    num_edges = len(edges)
    
    # Deadline guard: 90s wall-clock limit, use 80s for safety
    deadline = time.time() + 80.0
    
    # First, do random sampling to find a valid starting point
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = float('inf')
    
    # Try random samples
    num_random = min(budget, 10000)
    for i in range(num_random):
        if time.time() > deadline - 20.0:
            break
        
        signs = rng.choice([-1, 1], size=num_edges)
        radius, valid = _evaluate(n, edges, signs)
        
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: try all +1 (should be connected for prism graphs)
        best_signs = np.ones(num_edges, dtype=np.int8)
        radius, valid = _evaluate(n, edges, best_signs)
        if valid:
            best_radius = radius
        else:
            # Try all -1
            best_signs = -np.ones(num_edges, dtype=np.int8)
            radius, valid = _evaluate(n, edges, best_signs)
            if valid:
                best_radius = radius
            else:
                # This shouldn't happen for prism graphs, but handle gracefully
                best_signs = np.ones(num_edges, dtype=np.int8)
    
    # Now do local search from the best starting point
    if time.time() < deadline - 10.0:
        improved_signs, improved_radius = _local_search(
            n, edges, best_signs, 
            max_iter=10000,
            time_limit=min(60.0, deadline - time.time() - 10.0)
        )
        
        if improved_radius is not None and improved_radius < best_radius:
            best_signs = improved_signs
            best_radius = improved_radius
    
    # Final check and return
    final_radius, valid = _evaluate(n, edges, best_signs)
    if not valid:
        # Emergency fallback
        for fallback in [np.ones(num_edges, dtype=np.int8), -np.ones(num_edges, dtype=np.int8)]:
            r, v = _evaluate(n, edges, fallback)
            if v:
                best_signs = fallback
                break
    
    return best_signs.astype(np.int8)
# EVOLVE-BLOCK-END
