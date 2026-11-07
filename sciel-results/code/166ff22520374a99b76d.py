"""
Constructs a graph on 16 vertices with maximum edges such that 
diameter is exactly 2 and every edge is essential.

The optimal graph is the complete bipartite graph K_{8,8}.
- It has 64 edges.
- It is triangle-free, so removing any edge (u,v) removes the only path of length 1.
  Since u and v are in different partitions, they have no common neighbors.
  Thus, no path of length 2 exists between u and v after removal.
  The shortest path becomes length 3 (via another vertex in each partition).
  Diameter increases from 2 to 3.
  Thus, every edge is essential.
"""

def construct():
    """
    Returns the edges of K_{8,8} on vertices 0..15.
    Partition A: 0..7
    Partition B: 8..15
    """
    edges = []
    # Connect every vertex in A to every vertex in B
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    
    # Ensure types are Python int and list
    return [[int(u), int(v)] for u, v in edges]
