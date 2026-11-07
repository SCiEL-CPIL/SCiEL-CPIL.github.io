"""Rewritten program for spectral graph lifts task."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build adjacency for spanning tree
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    # BFS to find spanning tree
    visited = [False] * n
    tree_edges = set()
    queue = deque([0])
    visited[0] = True
    while queue:
        u = queue.popleft()
        for v, eidx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edges.add(eidx)
                queue.append(v)
    
    # Chord edges
    chord_edges = [i for i in range(num_edges) if i not in tree_edges]
    num_chords = len(chord_edges)
    
    # Precompute edge endpoints for fast matrix construction
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    
    # Base signs: +1 for tree, variable for chords
    # We will iterate masks for chords
    
    best_radius = float('inf')
    best_signs = None
    
    # If num_chords is 0 (should not happen for prism m>=4), handle
    if num_chords == 0:
        return np.ones(num_edges, dtype=np.int8)
        
    for mask in range(1 << num_chords):
        # The lift is disconnected iff all chord signs are +1 (mask == 0)
        # because tree signs are +1, making the graph "balanced"
        if mask == 0:
            continue
            
        # Construct signs array
        signs = np.ones(num_edges, dtype=np.int8)
        for i, chord_idx in enumerate(chord_edges):
            if (mask >> i) & 1:
                signs[chord_idx] = -1
            else:
                signs[chord_idx] = 1
        
        # Compute spectral radius
        # A[u,v] = signs[idx]
        A = np.zeros((n, n), dtype=np.float64)
        # Use vectorized assignment if possible, but loop is fine for small n
        for idx in range(num_edges):
            s = float(signs[idx])
            A[edge_u[idx], edge_v[idx]] = s
            A[edge_v[idx], edge_u[idx]] = s
            
        eigvals = np.linalg.eigvalsh(A)
        radius = np.max(np.abs(eigvals))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    if best_signs is None:
        # Fallback
        return np.ones(num_edges, dtype=np.int8)
        
    return best_signs
# EVOLVE-BLOCK-END
