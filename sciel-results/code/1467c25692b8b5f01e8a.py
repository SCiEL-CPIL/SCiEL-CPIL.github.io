"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    edges = []
    
    # First K_{3,3} on vertices 0-5
    # Partitions: {0, 1, 2} and {3, 4, 5}
    p1 = [0, 1, 2]
    p2 = [3, 4, 5]
    for u in p1:
        for v in p2:
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6-11
    # Partitions: {6, 7, 8} and {9, 10, 11}
    p3 = [6, 7, 8]
    p4 = [9, 10, 11]
    for u in p3:
        for v in p4:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
