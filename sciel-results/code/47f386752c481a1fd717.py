import time

def construct():
    """
    Constructs a graph on 12 vertices (0..11) consisting of two disjoint K_{3,3} components.
    Component 1: Vertices 0, 1, 2 (left) and 3, 4, 5 (right).
    Component 2: Vertices 6, 7, 8 (left) and 9, 10, 11 (right).
    
    Properties:
    - Simple, undirected.
    - Min degree 3.
    - Triangle-free (bipartite).
    - Girth 4.
    - Cycle lengths present: 4 and 6.
    - Distinct cycle lengths: {4, 6} -> Count = 2.
    """
    edges = []
    
    # Component 1: K_{3,3} on {0,1,2} x {3,4,5}
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on {6,7,8} x {9,10,11}
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    return edges
