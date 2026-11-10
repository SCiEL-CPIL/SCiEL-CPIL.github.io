def construct():
    n = 12
    edges = []
    # First K3,3 on vertices 0-5
    # Left: 0, 1, 2 ; Right: 3, 4, 5
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
            
    # Second K3,3 on vertices 6-11
    # Left: 6, 7, 8 ; Right: 9, 10, 11
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
            
    return edges
