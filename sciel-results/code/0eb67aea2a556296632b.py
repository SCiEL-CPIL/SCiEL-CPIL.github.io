import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge list
    edge_list = [(u, v) for u, v in edges]
    
    # Precompute adjacency for connectivity check
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edge_list):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Precompute edge types: cycle_top, cycle_bottom, matching
    edge_types = []
    for u, v in edge_list:
        if u < m and v < m:
            edge_types.append('top')
        elif u >= m and v >= m:
            edge_types.append('bottom')
        else:
            edge_types.append('match')
    
    def build_signed_matrix(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edge_list):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_matrix(signs)
        evals = np.linalg.eigvalsh(A)
        return np.max(np.abs(evals))
    
    def is_connected_lift(signs):
        # Check if signing is a coboundary using BFS
        t = np.zeros(n, dtype=int)
        t[0] = 1
        
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v, ei in adj[u]:
                if not visited[v]:
                    t[v] = t[u] * signs[ei]
                    visited[v] = True
                    queue.append(v)
                else:
                    if t[v] != t[u] * signs[ei]:
                        return True
        return False

    # Initial valid signing: all +1 is always connected?
    # For a prism graph, if all signs are +1, the lift has two components?
    # Let's check:
    # Vertices (v,0) and (v,1).
    # Edge (u,v) with sign +1 connects (u,0)-(v,0) and (u,1)-(v,1).
    # If the base graph is connected, the lift with all +1 signs has two components:
    # {(v,0) for all v} and {(v,1) for all v}.
    # So all +1 is INVALID (not connected).
    
    # We need a valid start.
    # A valid signing requires that the product of signs around any cycle is not necessarily -1,
    # but the lift is connected iff the signing is NOT a coboundary.
    # i.e., there exists a cycle with product -1.
    
    # Strategy: Start with all +1, then flip one edge on a cycle to make it -1.
    # This ensures connectivity.
    
    initial_signs = np.ones(num_edges, dtype=int)
    # Flip the first top cycle edge to -1
    # Find an edge on the top cycle
    top_edge_idx = -1
    for i, (u, v) in enumerate(edge_list):
        if u < m and v < m:
            top_edge_idx = i
            break
    if top_edge_idx != -1:
        initial_signs[top_edge_idx] = -1
    else:
        # Fallback: flip first edge
        initial_signs[0] = -1
        
    if not is_connected_lift(initial_signs):
        # Try flipping another edge
        for i in range(num_edges):
            initial_signs[i] = -initial_signs[i]
            if is_connected_lift(initial_signs):
                break
            initial_signs[i] = -initial_signs[i]

    best_signs = initial_signs.copy()
    best_radius = spectral_radius(best_signs)
    
    rng = np.random.default_rng(3)
    
    # Simulated Annealing
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    T_start = 0.5
    T_end = 0.001
    # Number of steps determined by time
    # Each step involves:
    # 1. Flip edge
    # 2. Check connectivity (fast)
    # 3. Compute radius (slow)
    
    # Estimate time per step
    # Let's run until deadline
    
    while time.time() < deadline - 2.0:
        # Adaptive temperature
        # We don't know total steps, so use time-based cooling
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        # Simple linear cooling based on remaining time?
        # Or just use a fixed number of iterations per "chunk"
        
        # Let's do a batch of 100 steps
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
                
            # Pick random edge
            i = rng.integers(0, num_edges)
            
            # Flip
            old_sign = current_signs[i]
            current_signs[i] = -old_sign
            
            # Check validity
            if is_connected_lift(current_signs):
                new_radius = spectral_radius(current_signs)
                
                # Accept if better
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_signs = current_signs.copy()
                
                # Accept if worse with probability exp(-delta/T)
                if new_radius > current_radius:
                    # Calculate T
                    # Use a simple schedule: T = T_start * (T_end/T_start)^(elapsed/total_time)
                    # Since we don't know total_time, let's just use a fixed low T for now
                    # or estimate based on remaining time.
                    # Let's use T = 0.1 * (remaining / 80.0) + 0.01
                    T = 0.1 * (remaining / 80.0) + 0.01
                    delta = new_radius - current_radius
                    if rng.random() < np.exp(-delta / T):
                        current_radius = new_radius
                    else:
                        current_signs[i] = old_sign
                else:
                    current_radius = new_radius
            else:
                # Invalid, revert
                current_signs[i] = old_sign
        
        # Occasionally, if stuck, do a random restart from best?
        # Or just keep going. SA should handle it.
        
        # Update remaining time for next loop iteration
        remaining = deadline - time.time()
        
    return best_signs
