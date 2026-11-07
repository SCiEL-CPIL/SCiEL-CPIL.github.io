"""Program: minimize distinct simple cycle lengths on 12 vertices with min degree 3, girth >= 4."""

import time

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 12 vertices (0..11) that minimizes the number of distinct
    simple cycle lengths.
    
    The optimal value is 2, achieved by two disjoint K_{3,3} components.
    K_{3,3} has cycle lengths {4, 6}.
    Two disjoint copies have cycle lengths {4, 6}.
    
    It is impossible to achieve value 1 with 12 vertices and min degree 3:
    - Value 1 requires all cycles to have the same length L.
    - If L=4, min degree 3 implies a 6-cycle exists (theorem).
    - If L=5, the smallest 3-regular girth-5 graph is Petersen (10 vertices). 
      We have 2 extra vertices. Any valid attachment creates new cycle lengths.
      Disjoint components of size 6 cannot have girth 5 (min size is 10).
    - If L=6, the smallest 3-regular girth-6 graph has 30 vertices (Moore bound).
    
    Thus, value 2 is the global minimum.
    """
    
    # Deadline guard: track elapsed time for safety
    t0 = time.time()
    deadline = t0 + 80.0  # 80s safety margin within 90s limit
    
    # Static construction of K_{3,3} union K_{3,3}
    edges = []
    
    # Component 1: K_{3,3} on {0,1,2} x {3,4,5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on {6,7,8} x {9,10,11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    # Type safety: ensure all elements are Python int, edges are lists
    result = []
    for e in edges:
        u, v = e
        result.append([int(u), int(v)])
        
    return result

# EVOLVE-BLOCK-END
