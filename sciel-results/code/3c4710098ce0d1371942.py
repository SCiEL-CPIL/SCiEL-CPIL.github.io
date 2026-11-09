"""Rewritten program for spectral graph lifts task."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_matrix(graph, signs):
    """Build the signed adjacency matrix for the base graph."""
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n))
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        A[u, v] = s
        A[v, u] = s
    return A

def _spectral_radius(A):
    """Compute the spectral radius (max absolute eigenvalue)."""
    evals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(evals)))

def _is_connected_lift(graph, signs):
    """
    Check if the 2-lift defined by signs is connected.
    
    The lift has vertices (v, 0) and (v, 1) for each base vertex v.
    An edge (u, v) with sign +1 connects (u,0)-(v,0) and (u,1)-(v,1).
    An edge (u, v) with sign -1 connects (u,0)-(v,1) and (u,1)-(v,0).
    
    We check connectivity via BFS on the 2n vertices.
    """
    n = graph['n']
    edges = graph['edges']
    N = 2 * n
    
    # Build adjacency for the lift
    adj = [[] for _ in range(N)]
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            # Same side connections
            adj[u * 2].append(v * 2)
            adj[v * 2].append(u * 2)
            adj[u * 2 + 1].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2 + 1)
        else:
            # Opposite side connections
            adj[u * 2].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2)
            adj[u * 2 + 1].append(v * 2)
            adj[v * 2].append(u * 2 + 1)
    
    # BFS from vertex 0
    visited = [False] * N
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                queue.append(neighbor)
    
    return count == N

def sign_edges(graph, budget):
    """
    Find signs for edges of a prism graph to minimize the spectral radius
    of the signed adjacency matrix, subject to the lift being connected.
    """
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    E = len(edges)
    
    # The prism graph has 3m edges: m cycle edges on top, m cycle edges on bottom, m matching edges
    # For m in {4, 5, 7, 9}, E is 12, 15, 21, 27
    
    # We'll use a randomized local search approach:
    # 1. Start with random valid configurations
    # 2. Flip individual signs and keep if it improves (or stays valid and improves)
    # 3. Try pairs of flips for more complex moves
    
    best_signs = None
    best_radius = float('inf')
    
    def evaluate(signs):
        """Evaluate a sign configuration, return radius or inf if invalid."""
        if not _is_connected_lift(graph, signs):
            return float('inf')
        A = _build_matrix(graph, signs)
        return _spectral_radius(A)
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    
    # First, try a systematic approach: enumerate some structured patterns
    # For prism graphs, there's a natural structure:
    # - Top cycle edges: (i, i+1) mod m for i in 0..m-1
    # - Bottom cycle edges: (m+i, m+i+1) mod m for i in 0..m-1  
    # - Matching edges: (i, m+i) for i in 0..m-1
    
    # Let's identify which edges are which type
    top_cycle = []
    bottom_cycle = []
    matching = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle.append(i)
        elif u >= m and v >= m:
            bottom_cycle.append(i)
        else:
            matching.append(i)
    
    # Strategy: Use local search with multiple restarts
    for restart in range(200):
        if time.time() > deadline - 5:
            break
            
        # Generate random initial signs
        signs = rng.choice([-1, 1], size=E)
        
        # Ensure connectivity, otherwise try a few random flips
        if not _is_connected_lift(graph, signs):
            # Try flipping some signs to fix connectivity
            fixed = False
            for _ in range(50):
                idx = rng.integers(0, E)
                signs[idx] = -signs[idx]
                if _is_connected_lift(graph, signs):
                    fixed = True
                    break
            if not fixed:
                continue
        
        # Local search: flip one sign at a time
        improved = True
        while improved:
            improved = False
            current_radius = evaluate(signs)
            if current_radius >= best_radius:
                # We can still try to improve, but if we're already worse than best, 
                # we need to be more selective
                pass
            
            for i in range(E):
                if time.time() > deadline - 5:
                    break
                    
                old_sign = signs[i]
                signs[i] = -old_sign
                
                new_radius = evaluate(signs)
                
                if new_radius < current_radius:
                    current_radius = new_radius
                    improved = True
                else:
                    signs[i] = old_sign
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_signs = signs.copy()
        
        # Try pair flips for more complex moves
        if best_signs is not None:
            signs = best_signs.copy()
            current_radius = best_radius
            
            for i in range(E):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, E):
                    if time.time() > deadline - 3:
                        break
                    
                    signs[i] = -signs[i]
                    signs[j] = -signs[j]
                    
                    new_radius = evaluate(signs)
                    
                    if new_radius < current_radius:
                        current_radius = new_radius
                        best_radius = new_radius
                        best_signs = signs.copy()
                    else:
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]
    
    # If no valid solution found, try a simple heuristic
    if best_signs is None:
        # All +1 signs give the original graph's adjacency
        # Check if that's connected (it should be for prism graphs)
        signs = np.ones(E, dtype=int)
        if _is_connected_lift(graph, signs):
            best_signs = signs
        else:
            # Fallback: try random until we find one
            for _ in range(1000):
                signs = rng.choice([-1, 1], size=E)
                if _is_connected_lift(graph, signs):
                    best_signs = signs
                    break
            if best_signs is None:
                # Last resort
                best_signs = np.ones(E, dtype=int)
    
    return best_signs
# EVOLVE-BLOCK-END
