"""Rewritten program for the suite task."""
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a complete bipartite graph K_{8,8} on vertices 0..15.
    Partitions: A = {0..7}, B = {8..15}.
    Edges connect every vertex in A to every vertex in B.
    
    Properties:
    1. Diameter is exactly 2.
    2. Triangle-free (necessary for edge-criticality in diameter 2).
    3. Deleting any edge (u,v) with u in A, v in B increases the distance 
       between u and v from 1 to 3 (since no path of length 2 exists between 
       partitions in a bipartite graph), thus increasing the diameter to 3.
    4. Edge count is 64, which is the maximum possible for a triangle-free 
       graph on 16 vertices (Turán's theorem).
    """
    edges = []
    # Partition A: 0 to 7
    # Partition B: 8 to 15
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
