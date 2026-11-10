def construct():
    edges = []
    # A = 0, 1, 2
    # B1 = 3, 4, 5
    # B2 = 6..15 (10 vertices)
    
    # Edges within A (K3)
    for i in range(3):
        for j in range(i+1, 3):
            edges.append([i, j])
            
    # Edges between A and B (all)
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
            
    # Edges between B1 and B2
    for b1 in range(3, 6):
        for b2 in range(6, 16):
            edges.append([b1, b2])
            
    return edges
