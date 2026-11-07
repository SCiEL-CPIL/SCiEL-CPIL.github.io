"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with 12 vertices, minimum degree 3, and girth at least 4.
    Uses two disjoint copies of the complete bipartite graph K_{3,3}.
    K_{3,3} has cycles of length 4 and 6 only.
    Thus, the number of distinct cycle lengths is 2.
    """
    edges = []
    
    # First K_{3,3} on vertices 0..5
    # Partition A: {0, 1, 2}, Partition B: {3, 4, 5}
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6..11
    # Partition A: {6, 7, 8}, Partition B: {9, 10, 11}
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
