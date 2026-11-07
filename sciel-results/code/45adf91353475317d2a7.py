import itertools

def construct():
    """
    Constructs a graph on 12 vertices (0..11) with min degree 3, girth >= 4,
    and exactly 2 distinct simple cycle lengths (4 and 6).
    The graph consists of two disjoint copies of K3,3.
    """
    
    # Component 1: K3,3 on vertices {0, 1, 2} and {3, 4, 5}
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    
    # Component 2: K3,3 on vertices {6, 7, 8} and {9, 10, 11}
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    
    edges = []
    
    # Generate edges for first K3,3
    for u, v in itertools.product(part_a1, part_b1):
        edges.append([u, v])
        
    # Generate edges for second K3,3
    for u, v in itertools.product(part_a2, part_b2):
        edges.append([u, v])
        
    return edges
