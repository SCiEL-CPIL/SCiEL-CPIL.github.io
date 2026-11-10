"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a graph on vertices 0..11 with:
    - Every vertex has degree at least 3
    - Every simple cycle has length at least 4 (triangle-free)
    - Minimize the number of distinct simple cycle lengths
    
    Strategy: Use two disjoint copies of K_{3,3}.
    K_{3,3} is 3-regular, bipartite (so girth 4, no triangles),
    and contains only cycles of length 4 and 6.
    Two disjoint copies give 12 vertices, all degree 3, girth 4,
    and only cycle lengths {4, 6} -> raw value = 2.
    
    This is believed to be the minimum possible value for this problem size
    and constraints, as 3-regular triangle-free graphs on 12 vertices
    must contain 4-cycles (due to Moore bound constraints on girth 6)
    and typically also 6-cycles.
    """
    # First copy of K_{3,3}: vertices 0-5
    # Left partition: {0, 1, 2}, Right partition: {3, 4, 5}
    edges = set()
    for i in range(3):
        for j in range(3):
            u = i
            v = 3 + j
            edges.add(tuple(sorted((u, v))))
    
    # Second copy of K_{3,3}: vertices 6-11
    # Left partition: {6, 7, 8}, Right partition: {9, 10, 11}
    for i in range(3):
        for j in range(3):
            u = 6 + i
            v = 9 + j
            edges.add(tuple(sorted((u, v))))
    
    # Convert to list of [u, v] lists
    result = [list(e) for e in sorted(edges)]
    
    return result

# EVOLVE-BLOCK-END
