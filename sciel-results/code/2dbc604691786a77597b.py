def construct():
    # Vertices 0..15
    # Partition A: 0, 1, 2
    # Partition B: 3..15
    # Edges:
    # 1. All edges between A and B (3 * 13 = 39 edges)
    # 2. All edges within A (3 choose 2 = 3 edges)
    # Total 42 edges.
    
    edges = []
    
    # Bipartite edges
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
            
    # Clique edges in A
    edges.append([0, 1])
    edges.append([0, 2])
    edges.append([1, 2])
    
    return edges
