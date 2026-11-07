def construct():
    edges = []
    
    # Define the two components. Each is a K_{3,3}.
    # Component 1: Vertices 0, 1, 2 and 3, 4, 5
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    
    # Component 2: Vertices 6, 7, 8 and 9, 10, 11
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    
    # Generate edges for Component 1
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Generate edges for Component 2
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    return edges
