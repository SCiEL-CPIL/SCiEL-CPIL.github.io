"""
Constructs a C5-free graph with independence number < 9.
The known Ramsey number R(C5, K9) is 33, which implies that the maximum
number of vertices in a C5-free graph with alpha < 9 is 32.
A construction achieving this is 8 disjoint copies of K4.
"""

def construct():
    # 8 disjoint copies of K4
    # Vertices 0..3 form K4
    # Vertices 4..7 form K4
    # ...
    # Vertices 28..31 form K4
    
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}
