def construct():
    # Grötzsch graph on vertices 0..10
    # Cycle 0-1-2-3-4-0
    edges = []
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    
    # Mycielski construction
    # Vertices 5..9 are copies of 0..4
    # Vertex 10 is the top
    for i in range(5):
        # Connect copy i to neighbors of i in cycle
        # Neighbors of i are (i-1)%5 and (i+1)%5
        edges.append([i, 5 + (i - 1) % 5])
        edges.append([i, 5 + (i + 1) % 5])
        # Connect top to copy i
        edges.append([10, 5 + i])
        
    # Independent set I = {5, 6, 7, 8, 9}
    I = [5, 6, 7, 8, 9]
    
    # Add vertices 11..15
    # Connect each to all vertices in I
    for v in range(11, 16):
        for u in I:
            edges.append([v, u])
            
    return edges
