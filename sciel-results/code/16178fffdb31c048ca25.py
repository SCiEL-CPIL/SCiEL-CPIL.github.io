import time

def construct():
    """
    Constructs the Clebsch graph on 16 vertices.
    The Clebsch graph is a triangle-free graph with chromatic number 4.
    It has 40 edges, which is the maximum for triangle-free graphs on 16 vertices
    that are not 3-colorable.
    
    Construction:
    1. Start with the 4-dimensional hypercube Q4 (vertices 0..15, edges if Hamming distance 1).
    2. Add a perfect matching connecting each vertex to its bitwise complement 
       (Hamming distance 4).
    """
    n = 16
    edges = []
    
    # 1. Add edges of the 4-cube (Hamming distance 1)
    for u in range(n):
        for v in range(u + 1, n):
            # Check Hamming distance
            xor = u ^ v
            # Count bits
            if bin(xor).count('1') == 1:
                edges.append([u, v])
                
    # 2. Add complement edges (Hamming distance 4)
    # Each vertex u is connected to ~u (which is (15-u) in 4-bit representation? 
    # No, ~u in 4 bits is (15 ^ u).
    # Since we iterate u < v, we just add if v == (15 ^ u)
    for u in range(n):
        v = 15 ^ u
        if u < v:
            edges.append([u, v])
            
    return edges
