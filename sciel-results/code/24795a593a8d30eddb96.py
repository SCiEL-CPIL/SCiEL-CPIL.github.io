import time

def construct():
    """
    Constructs a C5-free graph with independence number < 9.
    Uses 8 disjoint copies of K4.
    n = 32.
    alpha = 8.
    This certifies R(C5, K9) > 32.
    It is known that R(C5, K9) = 33, so 32 is the maximum n.
    """
    n = 32
    edges = []
    
    # 8 blocks of 4 vertices
    for b in range(8):
        base = b * 4
        # Add all edges within the K4
        # Vertices: base, base+1, base+2, base+3
        # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}
