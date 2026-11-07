import time
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph consisting of two disjoint copies of K_3,3.
    This graph is 3-regular, bipartite (triangle-free), and has 12 vertices.
    The simple cycles in K_3,3 have lengths 4 and 6.
    Therefore, the distinct cycle lengths are {4, 6}, minimizing the count to 2.
    """
    edges = []
    
    # First K_3,3 component on vertices 0..5
    # Part A: 0, 1, 2
    # Part B: 3, 4, 5
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
            
    # Second K_3,3 component on vertices 6..11
    # Part A: 6, 7, 8
    # Part B: 9, 10, 11
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
            
    # Ensure types are correct (integers in lists)
    # The loop variables u and v are already Python ints.
    # edges is a list of lists of ints.
    
    return edges

# EVOLVE-BLOCK-END
