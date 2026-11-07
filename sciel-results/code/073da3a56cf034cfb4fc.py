import time

def construct():
    """
    Constructs a graph on 12 vertices (0..11) that minimizes the number of distinct
    simple cycle lengths.
    
    The optimal construction is two disjoint K_{3,3} components.
    Properties:
    - Simple, undirected, bipartite.
    - Minimum degree = 3.
    - Girth = 4.
    - Cycle lengths present: 4 and 6 only.
    - Distinct cycle lengths count: 2.
    
    This is optimal because:
    1. Value 1 is impossible for min-degree-3 triangle-free graphs on 12 vertices.
       - Girth 4: Bipartite graphs with min degree 3 on 12 vertices must contain 6-cycles.
       - Girth 5: The Petersen graph (10 vertices) is the smallest 3-regular girth-5 graph.
         Extending it to 12 vertices while maintaining min degree 3 and girth 5 is impossible
         without introducing cycles of other lengths.
       - Girth 6: The Moore bound for degree 3 and girth 6 requires at least 14 vertices.
    2. The K_{3,3} union achieves value 2.
    """
    
    # Construct edges for two disjoint K_{3,3} components
    edges = []
    
    # Component 1: K_{3,3} on vertices {0, 1, 2} and {3, 4, 5}
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices {6, 7, 8} and {9, 10, 11}
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    # Ensure types are Python int and structure is list of lists
    result = []
    for e in edges:
        u, v = e
        result.append([int(u), int(v)])
        
    return result
