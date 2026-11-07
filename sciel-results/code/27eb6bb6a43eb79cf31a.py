"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 12 vertices (0..11) consisting of two disjoint K_{3,3} components.
    
    Properties:
    - Undirected simple graph.
    - Vertices: 0 to 11.
    - Component 1: K_{3,3} with partitions {0, 1, 2} and {3, 4, 5}.
    - Component 2: K_{3,3} with partitions {6, 7, 8} and {9, 10, 11}.
    - Minimum degree: 3 (each vertex in K_{3,3} has degree 3).
    - Girth: 4 (bipartite graphs have no odd cycles, smallest cycle in K_{3,3} is 4).
    - Distinct cycle lengths: {4, 6}.
    - Count of distinct cycle lengths: 2.
    """
    edges = []
    
    # Component 1: K_{3,3}
    # Left partition: 0, 1, 2
    # Right partition: 3, 4, 5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Component 2: K_{3,3}
    # Left partition: 6, 7, 8
    # Right partition: 9, 10, 11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
