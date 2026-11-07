def sign_edges(graph, budget):
    import numpy as np
    
    n = graph['n']
    edges = list(graph['edges'])
    E = len(edges)
    m = n // 2
    
    # Map edge (u, v) to index
    # Ensure consistent ordering for keys
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        if u > v:
            u, v = v, u
        edge_idx[(u, v)] = i
        
    # Identify specific edges
    # 1. Matching edges (i, i+m)
    match_indices = []
    for i in range(m):
        u, v = i, i + m
        if u > v: u, v = v, u
        match_indices.append(edge_idx[(u, v)])
        
    # 2. Top cycle edges (i, (i+1)%m)
    top_edges = []
    for i in range(m):
        u, v = i, (i + 1) % m
        if u > v: u, v = v, u
        top_edges.append(edge_idx[(u, v)])
        
    # 3. Bottom cycle edges (m+i, m+(i+1)%m)
    bot_edges = []
    for i in range(m):
        u, v = m + i, m + (i + 1) % m
        if u > v: u, v = v, u
        bot_edges.append(edge_idx[(u, v)])
        
    # Define Tree and Chords
    # Tree: All matching edges + Top path edges (0 to m-2)
    # Chords: Top close edge (m-1 to 0) + All bottom edges
    
    tree_indices = set()
    for idx in match_indices:
        tree_indices.add(idx)
    for i in range(m - 1):
        tree_indices.add(top_edges[i])
        
    chord_indices = []
    # Top close is top_edges[m-1]
    chord_indices.append(top_edges[m - 1])
    # Bottom edges
    for idx in bot_edges:
        chord_indices.append(idx)
        
    num_chords = len(chord_indices) # Should be m + 1
    
    best_radius = float('inf')
    best_signs = None
    
    # Enumerate masks 1 to 2^num_chords - 1
    # Mask 0 is balanced (disconnected), so skip it.
    for mask in range(1, 1 << num_chords):
        signs = np.ones(E, dtype=np.int8)
        
        # Set chord signs based on mask
        for j, cidx in enumerate(chord_indices):
            if (mask >> j) & 1:
                signs[cidx] = -1
            else:
                signs[cidx] = 1
                
        # Tree edges are already +1
        
        # Build signed adjacency matrix
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
            
        vals = np.linalg.eigvalsh(A)
        radius = np.max(np.abs(vals))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    if best_signs is None:
        # Fallback, should not happen for m>=2
        best_signs = np.ones(E, dtype=np.int8)
        best_signs[0] = -1
        
    return best_signs.astype(np.int8)
