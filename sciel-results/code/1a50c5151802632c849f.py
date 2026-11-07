import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build edge lookup
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
    
    # Build adjacency list for faster traversal
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Identify spanning tree and chords using BFS
    # Tree edges: all matching edges + all top cycle edges except closing edge
    # Chords: top cycle closing edge + all bottom cycle edges
    
    # Build a proper spanning tree:
    # Start with vertex 0, do BFS
    visited = [False] * n
    tree_edges = set()
    visited[0] = True
    queue = [0]
    
    while queue:
        u = queue.pop(0)
        for v, ei in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edges.add(ei)
                queue.append(v)
    
    # Chords are edges not in the spanning tree
    chord_indices = [i for i in range(num_edges) if i not in tree_edges]
    num_chords = len(chord_indices)
    
    # Precompute which edges are tree vs chord for quick sign assignment
    is_tree = np.zeros(num_edges, dtype=bool)
    for i in tree_edges:
        is_tree[i] = True
    
    best_radius = float('inf')
    best_signs = None
    
    # Helper: check if lift is connected (has a negative cycle)
    def is_lift_connected(signs):
        # BFS to check for parity inconsistency
        parity = [-1] * n
        parity[0] = 0
        queue = [0]
        
        while queue:
            u = queue.pop(0)
            for v, ei in adj[u]:
                s = signs[ei]
                expected_parity = parity[u] if s == 1 else 1 - parity[u]
                
                if parity[v] == -1:
                    parity[v] = expected_parity
                    queue.append(v)
                else:
                    if parity[v] != expected_parity:
                        return True
        
        return False
    
    # Precompute base signed matrix template (tree edges fixed to +1)
    # For each chord assignment, we only need to update chord edges
    
    # Iterate over all chord assignments
    for mask in range(1 << num_chords):
        if time.time() > deadline:
            break
            
        signs = np.ones(num_edges, dtype=np.int8)
        
        for j, idx in enumerate(chord_indices):
            if mask & (1 << j):
                signs[idx] = -1
        
        # Check validity
        if not is_lift_connected(signs):
            continue
            
        # Compute spectral radius efficiently
        # Build signed adjacency matrix
        A_signed = np.zeros((n, n), dtype=np.float64)
        for u, v in edges:
            idx = edge_to_idx[(u, v)]
            val = float(signs[idx])
            A_signed[u, v] = val
            A_signed[v, u] = val
            
        # Spectral radius
        eigvals = np.linalg.eigvalsh(A_signed)
        radius = np.max(np.abs(eigvals))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: flip first chord
        signs = np.ones(num_edges, dtype=np.int8)
        if num_chords > 0:
            signs[chord_indices[0]] = -1
        else:
            signs[0] = -1
        return signs
        
    return best_signs
