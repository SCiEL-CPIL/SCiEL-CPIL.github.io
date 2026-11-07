import numpy as np
import time

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Build adjacency list with edge indices
    adj = {i: [] for i in range(n)}
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        
    # Find a spanning tree using BFS
    visited = [False] * n
    tree_edges = set()
    queue = [0]
    visited[0] = True
    
    while queue:
        u = queue.pop(0)
        for v, idx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edges.add(idx)
                queue.append(v)
                
    # Identify non-tree edges
    non_tree_edges = [i for i in range(num_edges) if i not in tree_edges]
    k = len(non_tree_edges)
    
    best_radius = float('inf')
    best_signs = None
    
    # Pre-compute base adjacency matrix with all +1
    # Tree edges are always +1, non-tree edges vary
    # We can pre-compute the contribution of tree edges
    
    # Create a base matrix with tree edges set to 1
    base_A = np.zeros((n, n))
    for idx, (u, v) in enumerate(edges):
        if idx in tree_edges:
            base_A[u, v] = 1
            base_A[v, u] = 1
            
    # Pre-compute the non-tree edge endpoints
    non_tree_u = [edges[idx][0] for idx in non_tree_edges]
    non_tree_v = [edges[idx][1] for idx in non_tree_edges]
    
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    # Iterate over all 2^k combinations
    for mask in range(1 << k):
        if time.time() > deadline:
            break
            
        # Build signed adjacency matrix
        A = base_A.copy()
        
        # Set non-tree edge signs based on mask
        for i, edge_idx in enumerate(non_tree_edges):
            if (mask >> i) & 1:
                s = -1
            else:
                s = 1
            u = non_tree_u[i]
            v = non_tree_v[i]
            A[u, v] = s
            A[v, u] = s
            
        # Compute spectral radius
        eigvals = np.linalg.eigvalsh(A)
        radius = np.max(np.abs(eigvals))
        
        if radius < best_radius:
            best_radius = radius
            # Reconstruct full signs array
            signs = np.ones(num_edges, dtype=int)
            for i, edge_idx in enumerate(non_tree_edges):
                if (mask >> i) & 1:
                    signs[edge_idx] = -1
                else:
                    signs[edge_idx] = 1
            best_signs = signs
            
    # Fallback if no valid lift found
    if best_signs is None:
        signs = np.ones(num_edges, dtype=int)
        if num_edges > 0:
            signs[0] = -1
        best_signs = signs
        
    return best_signs
