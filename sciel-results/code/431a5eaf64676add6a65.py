"""Rewritten program to minimize distinct cycle lengths."""

def construct():
    """
    Constructs a graph on 12 vertices consisting of two disjoint K_{3,3} components.
    Vertices 0-5 form the first K_{3,3} with partitions {0, 2, 4} and {1, 3, 5}.
    Vertices 6-11 form the second K_{3,3} with partitions {6, 8, 10} and {7, 9, 11}.
    
    Properties:
    - Simple undirected graph.
    - Every vertex has degree 3.
    - Triangle-free (bipartite).
    - Contains cycles (lengths 4 and 6).
    - Distinct cycle lengths are exactly {4, 6}.
    """
    edges = []
    
    # First K_{3,3} component: vertices 0 to 5
    # Partition A: 0, 2, 4
    # Partition B: 1, 3, 5
    part_a_1 = [0, 2, 4]
    part_b_1 = [1, 3, 5]
    for u in part_a_1:
        for v in part_b_1:
            edges.append([u, v])
            
    # Second K_{3,3} component: vertices 6 to 11
    # Partition A: 6, 8, 10
    # Partition B: 7, 9, 11
    part_a_2 = [6, 8, 10]
    part_b_2 = [7, 9, 11]
    for u in part_a_2:
        for v in part_b_2:
            edges.append([u, v])

    # Return edges as list of lists of ints
    return edges
