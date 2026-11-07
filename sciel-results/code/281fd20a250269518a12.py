import time
import sys

def construct():
    """
    Constructs a graph on 12 vertices with minimum degree 3 and girth at least 4.
    The goal is to minimize the number of distinct simple cycle lengths.
    
    Analysis:
    - A connected graph with minimum degree 3 must have cycles of at least two distinct lengths.
    - The smallest such component is K_{3,3} (6 vertices), which has cycle lengths {4, 6}.
    - Using two disjoint K_{3,3} components on 12 vertices results in cycle lengths {4, 6}.
    - It is not possible to have a graph with only one distinct cycle length (e.g., only 4-cycles) 
      with minimum degree 3 on 12 vertices, as connected components with min degree 3 
      necessarily contain cycles of multiple lengths (e.g., 4 and 6 in K_{3,3}).
    - Thus, the minimum number of distinct cycle lengths is 2.
    
    This implementation returns the edges for two disjoint K_{3,3} graphs.
    """
    edges = []
    
    # Component 1: K_{3,3} on vertices 0,1,2 (left) and 3,4,5 (right)
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6,7,8 (left) and 9,10,11 (right)
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    # Ensure types are Python int and list
    final_edges = []
    for e in edges:
        final_edges.append([int(e[0]), int(e[1])])
        
    return final_edges
