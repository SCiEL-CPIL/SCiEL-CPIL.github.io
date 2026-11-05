def construct():
    # Base graph: K_{3,13}
    # Vertices 0, 1, 2 in A; 3..15 in B
    edges = []
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
    
    # Add edges within B to increase count
    # A matching in B is safe because Q3 requires more complex B-B connectivity
    # when combined with the A-B connections.
    # Add a matching of size 3: (3,4), (5,6), (7,8)
    edges.append([3, 4])
    edges.append([5, 6])
    edges.append([7, 8])
    
    return edges
