def construct():
    # Grötzsch graph on vertices 0..10
    # Cycle 0-1-2-3-4-0
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    
    # Mycielski construction details:
    # Vertices 5..9 are copies of 0..4
    # Vertex 10 is the apex
    # Copy i (vertex 5+i) is connected to neighbors of i in the cycle
    # Neighbors of 0 are 1, 4 -> 5 connected to 1, 4
    # Neighbors of 1 are 0, 2 -> 6 connected to 0, 2
    # Neighbors of 2 are 1, 3 -> 7 connected to 1, 3
    # Neighbors of 3 are 2, 4 -> 8 connected to 2, 4
    # Neighbors of 4 are 3, 0 -> 9 connected to 3, 0
    
    # Add edges from copies to cycle vertices
    edges += [[5, 1], [5, 4]]
    edges += [[6, 0], [6, 2]]
    edges += [[7, 1], [7, 3]]
    edges += [[8, 2], [8, 4]]
    edges += [[9, 3], [9, 0]]
    
    # Add edges from apex to copies
    edges += [[10, 5], [10, 6], [10, 7], [10, 8], [10, 9]]
    
    # The set of copy vertices {5, 6, 7, 8, 9} is an independent set in the Grötzsch graph.
    # We add 5 more vertices: 11, 12, 13, 14, 15.
    # Connect each of these to all vertices in the independent set {5, 6, 7, 8, 9}.
    # This adds 5 * 5 = 25 edges.
    # No triangles are formed because:
    # 1. No edges among 11-15.
    # 2. {5..9} is independent, so no triangles involving two vertices from {5..9}.
    # 3. 11-15 are not connected to each other.
    
    independent_set = [5, 6, 7, 8, 9]
    for v in range(11, 16):
        for u in independent_set:
            edges.append([v, u])
            
    return edges
