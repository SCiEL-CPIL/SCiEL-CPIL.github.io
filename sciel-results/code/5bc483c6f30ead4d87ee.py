"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 15}

def construct():
    n = P['n']
    # Partition: A = {0,1,2} (degree 8), B = {3,4,5,6,7,8,9,10} (degree 7), C = {11,12,13,14} (degree 6)
    # Valid 3-edge paths must be s-a-b-t with a in A, b in B (or vice versa), s in B, t in C (or vice versa).
    # This ensures deg(s) != deg(t) since all degrees in B are 7, in C are 6, etc.
    A = [0, 1, 2]
    B = [3, 4, 5, 6, 7, 8, 9, 10]
    C = [11, 12, 13, 14]
    
    edges = []
    # Edges between A and B (3 * 8 = 24 edges)
    for a in A:
        for b in B:
            edges.append([a, b])
    # Edges between B and C (8 * 4 = 32 edges)
    for b in B:
        for c in C:
            edges.append([b, c])
    # Edges between A and C (3 * 4 = 12 edges)
    for a in A:
        for c in C:
            edges.append([a, c])
    
    return edges
# EVOLVE-BLOCK-END
