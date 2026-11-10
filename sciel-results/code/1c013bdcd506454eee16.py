def construct():
    """
    Constructs a graph on 12 vertices (0..11) with minimum degree 3,
    triangle-free (girth >= 4), minimizing the number of distinct cycle lengths.
    
    Strategy: Use two disjoint K_3,3 components.
    K_3,3 is bipartite (no triangles).
    Cycles in K_3,3 are of length 4 and 6.
    Distinct cycle lengths: {4, 6}. Count = 2.
    Min degree in K_3,3 is 3.
    This satisfies all constraints.
    """
    edges = []
    
    # Component 1: K_3,3 on vertices 0,1,2 and 3,4,5
    part1_a = [0, 1, 2]
    part1_b = [3, 4, 5]
    for u in part1_a:
        for v in part1_b:
            edges.append([u, v])
            
    # Component 2: K_3,3 on vertices 6,7,8 and 9,10,11
    part2_a = [6, 7, 8]
    part2_b = [9, 10, 11]
    for u in part2_a:
        for v in part2_b:
            edges.append([u, v])
    
    return edges
