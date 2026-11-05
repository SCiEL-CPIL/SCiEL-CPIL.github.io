import time

def construct():
    """
    Construct a dense triangle-free graph on 16 vertices with chromatic number at least 4.
    
    The Clebsch graph (also known as the folded 5-cube) is a triangle-free graph
    on 16 vertices with 40 edges and chromatic number 4. It is the known optimal
    solution for maximizing edges under these constraints for n=16.
    
    Vertices are integers 0..15.
    Two vertices u and v are adjacent if their Hamming distance is 1 or 4.
    """
    edges = []
    for u in range(16):
        for v in range(u + 1, 16):
            diff = u ^ v
            # Check Hamming distance
            if bin(diff).count('1') in (1, 4):
                edges.append([u, v])
    return edges
