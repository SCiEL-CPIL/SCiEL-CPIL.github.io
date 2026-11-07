"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 12 vertices consisting of two disjoint K_{3,3} components.
    Each K_{3,3} has minimum degree 3 and girth 4.
    The simple cycles in K_{3,3} have lengths 4 and 6.
    Thus, the set of distinct cycle lengths is {4, 6}, which has size 2.
    This minimizes the number of distinct cycle lengths for this vertex count and degree constraint.
    """
    edges = []
    
    # First K_{3,3} component: vertices 0, 1, 2 connected to 3, 4, 5
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Second K_{3,3} component: vertices 6, 7, 8 connected to 9, 10, 11
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
