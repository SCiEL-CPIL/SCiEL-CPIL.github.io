"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    """
    Construct an undirected simple graph on vertices 0..11.
    
    Design: Two disjoint copies of K_{3,3}.
    - Each K_{3,3} is 3-regular, bipartite, girth 4.
    - Cycle lengths in K_{3,3}: only 4 and 6.
    - Two disjoint components: cycle lengths remain {4, 6}.
    - Distinct cycle lengths: 2.
    """
    start_time = time.time()
    deadline = start_time + 85.0

    edges = []

    # Component 1: vertices 0,1,2 (part A) and 3,4,5 (part B)
    # Complete bipartite K_{3,3}
    for a in range(3):
        for b in range(3, 6):
            edges.append([a, b])

    # Component 2: vertices 6,7,8 (part A) and 9,10,11 (part B)
    # Complete bipartite K_{3,3}
    for a in range(6, 9):
        for b in range(9, 12):
            edges.append([a, b])

    return edges

# EVOLVE-BLOCK-END
