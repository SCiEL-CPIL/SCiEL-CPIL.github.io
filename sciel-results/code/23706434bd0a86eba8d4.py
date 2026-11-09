import time

def construct():
    # The Clebsch graph is a triangle-free graph on 16 vertices with 40 edges
    # and chromatic number 4. It is the unique strongly regular graph with
    # parameters (16, 5, 0, 2).
    #
    # Construction:
    # Vertices are the 16 binary strings of length 4 (integers 0 to 15).
    # Two vertices u and v are adjacent if their Hamming distance is 1 or 4.
    #
    # Hamming distance 1: 4 neighbors for each vertex.
    # Hamming distance 4: 1 neighbor for each vertex (the bitwise complement).
    # Total degree: 5.
    # Total edges: (16 * 5) / 2 = 40.
    #
    # This graph is triangle-free and 4-chromatic, which is the maximum
    # number of edges possible for a triangle-free 4-chromatic graph on 16 vertices.

    edges = []
    n = 16
    
    for u in range(n):
        for v in range(u + 1, n):
            # Calculate Hamming distance
            xor = u ^ v
            # Count bits set in xor
            dist = bin(xor).count('1')
            
            if dist == 1 or dist == 4:
                edges.append([u, v])
                
    return edges
