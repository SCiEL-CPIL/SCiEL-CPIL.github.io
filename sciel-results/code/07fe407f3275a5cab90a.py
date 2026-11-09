import numpy as np
import time
from itertools import combinations

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit (5s buffer)
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    e_count = len(edges)
    
    # Identify edge types for efficient processing
    # The prism graph has:
    # - m "top" cycle edges: (i, i+1) for i in 0..m-2, and (m-1, 0)
    # - m "bottom" cycle edges: (m+i, m+i+1) for i in 0..m-2, and (m+m-1, m)
    # - m "matching" edges: (i, i+m) for i in 0..m-1
    
    # Build adjacency for spanning tree
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Find spanning tree using BFS
    visited = [False] * n
    tree_edge_indices = set()
    queue = [0]
    visited[0] = True
    
    while queue:
        u = queue.pop(0)
        for v, edge_idx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edge_indices.add(edge_idx)
                queue.append(v)
                
    # Chords are edges not in the spanning tree
    chords = [i for i in range(e_count) if i not in tree_edge_indices]
    num_chords = len(chords)
    
    best_radius = float('inf')
    best_signs = np.ones(e_count, dtype=int)
    
    # Helper: compute spectral radius from signs
    def compute_radius(signs):
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    # Phase 1: Exhaustive search over chord assignments (tree edges = +1)
    # For m=4: 5 chords -> 32 masks
    # For m=5: 6 chords -> 64 masks
    # For m=7: 8 chords -> 256 masks
    # For m=9: 10 chords -> 1024 masks
    
    for mask in range(1 << num_chords):
        if time.time() > deadline:
            break
            
        signs = np.ones(e_count, dtype=int)
        
        for k, edge_idx in enumerate(chords):
            if mask & (1 << k):
                signs[edge_idx] = -1
        
        radius = compute_radius(signs)
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # Phase 2: 2-opt local search - flip pairs of edges to escape local minima
    # Start from the best solution found in Phase 1
    current_signs = best_signs.copy()
    current_radius = best_radius
    improved = True
    
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try all pairs of edge flips (2-opt moves)
        for i in range(e_count):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, e_count):
                if time.time() > deadline - 5:
                    break
                
                # Flip both edges i and j
                old_si = current_signs[i]
                old_sj = current_signs[j]
                current_signs[i] = -old_si
                current_signs[j] = -old_sj
                
                radius = compute_radius(current_signs)
                
                if radius < current_radius - 1e-12:
                    current_radius = radius
                    improved = True
                else:
                    # Revert
                    current_signs[i] = old_si
                    current_signs[j] = old_sj
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()
    
    # Phase 3: Single-flip polish pass
    if time.time() < deadline - 2:
        current_signs = best_signs.copy()
        current_radius = best_radius
        improved = True
        
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(e_count):
                if time.time() > deadline - 2:
                    break
                
                old_si = current_signs[i]
                current_signs[i] = -old_si
                
                radius = compute_radius(current_signs)
                
                if radius < current_radius - 1e-12:
                    current_radius = radius
                    improved = True
                else:
                    current_signs[i] = old_si
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_signs = current_signs.copy()
    
    return best_signs
