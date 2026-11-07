"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs an undirected simple graph on vertices 0..11.
    
    The graph consists of two disjoint copies of the complete bipartite graph K_{3,3}.
    - Component 1: Vertices 0,1,2 (part A) and 3,4,5 (part B).
    - Component 2: Vertices 6,7,8 (part A) and 9,10,11 (part B).
    
    Properties:
    - Every vertex has degree 3 (satisfies >= 3).
    - The graph is bipartite, so it is triangle-free (girth >= 4).
    - The simple cycles in K_{3,3} have lengths 4 and 6.
    - Thus, the set of distinct simple cycle lengths is {4, 6}.
    - The number of distinct cycle lengths is 2.
    
    Returns:
    - A list of edges [u, v] where u and v are Python ints.
    """
    edges = []
    
    # Component 1: K_{3,3} on vertices 0,1,2 (left) and 3,4,5 (right)
    part_a_1 = [0, 1, 2]
    part_b_1 = [3, 4, 5]
    for u in part_a_1:
        for v in part_b_1:
            edges.append([int(u), int(v)])
            
    # Component 2: K_{3,3} on vertices 6,7,8 (left) and 9,10,11 (right)
    part_a_2 = [6, 7, 8]
    part_b_2 = [9, 10, 11]
    for u in part_a_2:
        for v in part_b_2:
            edges.append([int(u), int(v)])
            
    return edges

# EVOLVE-BLOCK-END
