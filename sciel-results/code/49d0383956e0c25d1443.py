"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """Construct two disjoint K_{3,3} graphs on vertices 0..11.
    
    Component 1: vertices 0,1,2 (left) and 3,4,5 (right)
    Component 2: vertices 6,7,8 (left) and 9,10,11 (right)
    
    Each K_{3,3} is 3-regular, bipartite, triangle-free, and has
    only 4-cycles. Thus all simple cycles have length 4.
    Distinct cycle lengths = {4} -> count = 1.
    """
    edges = []
    
    # First K_{3,3}: left {0,1,2}, right {3,4,5}
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
    
    # Second K_{3,3}: left {6,7,8}, right {9,10,11}
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
    
    # Ensure all values are Python int
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
