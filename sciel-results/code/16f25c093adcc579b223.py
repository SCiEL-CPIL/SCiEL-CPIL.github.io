def construct():
    edges = []
    
    # First K3,3 on vertices 0..5
    # A = [0, 1, 2], B = [3, 4, 5]
    A1 = [0, 1, 2]
    B1 = [3, 4, 5]
    for u in A1:
        for v in B1:
            edges.append([u, v])
            
    # Second K3,3 on vertices 6..11
    # A = [6, 7, 8], B = [9, 10, 11]
    A2 = [6, 7, 8]
    B2 = [9, 10, 11]
    for u in A2:
        for v in B2:
            edges.append([u, v])
            
    return edges
