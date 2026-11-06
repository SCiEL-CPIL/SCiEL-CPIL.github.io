import time

def construct():
    """
    Construct a C5-free graph with alpha < 9, maximizing n.
    The best known construction is 8 disjoint K4s, which gives n=32 and alpha=8.
    """
    n = 32
    edges = []
    
    # Create 8 disjoint K4s
    # Vertices 0-3: K4
    # Vertices 4-7: K4
    # ...
    for block in range(8):
        start = block * 4
        # Complete graph on 4 vertices
        for i in range(4):
            for j in range(i + 1, 4):
                u = start + i
                v = start + j
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}
