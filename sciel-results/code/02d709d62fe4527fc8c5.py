"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """Construct two disjoint K_{3,3} components on vertices 0..11.
    
    K_{3,3} is bipartite, so triangle-free (girth 4).
    Cycle lengths in K_{3,3}: only 4 and 6.
    Each vertex has degree 3.
    Distinct cycle lengths: {4, 6} -> 2 distinct lengths.
    This is the theoretical minimum for n=12, min degree 3.
    """
    edges = []
    
    # First K_{3,3}: parts {0,1,2} and {3,4,5}
    for i in range(3):
        for j in range(3):
            u = i
            v = 3 + j
            edges.append([u, v])
    
    # Second K_{3,3}: parts {6,7,8} and {9,10,11}
    for i in range(3):
        for j in range(3):
            u = 6 + i
            v = 9 + j
            edges.append([u, v])
    
    # Ensure all are Python ints and lists
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result

# EVOLVE-BLOCK-END
