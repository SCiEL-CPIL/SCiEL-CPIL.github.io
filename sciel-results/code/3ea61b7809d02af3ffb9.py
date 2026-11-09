"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The seed solution consists of two disjoint K_{3,3} components.
    # K_{3,3} is bipartite with parts of size 3.
    # All cycles in K_{3,3} are of length 4 or 6.
    # Thus, the set of distinct cycle lengths is {4, 6}, count = 2.
    # It is conjectured that no graph on 12 vertices with min degree 3 
    # and girth >= 4 can have only 1 distinct cycle length.
    
    edges = []
    # Component 1: K_{3,3} on vertices 0..5
    A1 = [0, 1, 2]
    B1 = [3, 4, 5]
    for u in A1:
        for v in B1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6..11
    A2 = [6, 7, 8]
    B2 = [9, 10, 11]
    for u in A2:
        for v in B2:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
