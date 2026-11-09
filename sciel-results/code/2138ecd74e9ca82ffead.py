import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    
    # Build adjacency list
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
                
    # Identify chords (edges not in tree)
    chords = []
    for i in range(len(edges)):
        if i not in tree_edge_indices:
            chords.append(i)
            
    num_chords = len(chords)
    
    best_radius = float('inf')
    best_signs = None
    
    # Iterate over all masks
    # Mask bit k corresponds to chords[k]
    # 0 -> +1, 1 -> -1
    # We skip mask 0 because it results in a balanced (disconnected lift) graph
    for mask in range(1, 1 << num_chords): 
        if time.time() > deadline:
            if best_signs is not None:
                return best_signs
            # Fallback to a valid sign pattern if no time left and nothing found
            # Mask 1 is valid (one chord is -1, rest +1)
            signs = np.ones(len(edges), dtype=int)
            signs[chords[0]] = -1
            return signs
                
        signs = np.ones(len(edges), dtype=int)
        
        # Assign signs to chords
        for k, edge_idx in enumerate(chords):
            if mask & (1 << k):
                signs[edge_idx] = -1
            else:
                signs[edge_idx] = 1
                
        # Tree edges are already +1 (initialized)
        
        # Build signed adjacency matrix
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
            
        # Compute spectral radius
        eigvals = np.linalg.eigvalsh(A)
        radius = np.max(np.abs(eigvals))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    return best_signs
