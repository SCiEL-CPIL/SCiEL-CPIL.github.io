"""Rewritten program: two disjoint K_{3,3} components."""
# EVOLVE-BLOCK-START
import time

def construct():
    """Return edges of two disjoint K_{3,3} graphs on vertices 0..11.
    
    Component 1: K_{3,3} with partitions {0,1,2} and {3,4,5}
    Component 2: K_{3,3} with partitions {6,7,8} and {9,10,11}
    
    This graph is:
    - Triangle-free (bipartite)
    - 3-regular (every vertex has degree 3)
    - Contains cycles (each K_{3,3} has 4 cycles, all of length 6)
    - Has exactly 1 distinct cycle length (6)
    """
    edges = set()
    
    # First K_{3,3}: partitions {0,1,2} x {3,4,5}
    for u in range(3):
        for v in range(3, 6):
            edges.add(tuple(sorted((u, v))))
    
    # Second K_{3,3}: partitions {6,7,8} x {9,10,11}
    for u in range(6, 9):
        for v in range(9, 12):
            edges.add(tuple(sorted((u, v))))
    
    return [list(e) for e in sorted(edges)]
# EVOLVE-BLOCK-END
