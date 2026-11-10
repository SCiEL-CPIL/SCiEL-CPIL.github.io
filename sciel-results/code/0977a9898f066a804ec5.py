"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

def construct():
    """
    Constructs a simple undirected graph on 16 vertices with diameter exactly 2.
    Every edge is essential (critical) for maintaining diameter 2.
    
    The graph constructed is the complete bipartite graph K_{8,8}.
    The partitions are defined by the Most Significant Bit (MSB) of the vertex index.
    Partition A: vertices 0-7 (MSB is 0)
    Partition B: vertices 8-15 (MSB is 1)
    
    In K_{n/2, n/2}, the diameter is 2. 
    Removing any edge (u, v) disconnects u and v from their direct link.
    Since the graph is bipartite and complete between partitions, any u in A and v in B 
    have a distance of 1 if the edge exists, and distance 3 if it doesn't (u-AnyB-v... wait.
    If edge (u,v) is removed, dist(u,v) becomes 3 via u-w-v where w is another node in the 
    opposite partition? No.
    Let u in A, v in B.
    If (u,v) removed:
    Path u -> w -> v. 
    u is connected to all w in B.
    v is connected to all w in A.
    So u -> w_B -> v? No, w_B is in B. v is in B. No edge between B nodes.
    Path must be u (in A) -> w (in B) -> x (in A) -> v (in B)?
    Wait.
    If (u,v) is removed, u and v are still connected via other nodes.
    u is connected to all w in B (except v).
    v is connected to all x in A (except u).
    So u -> w (in B) -> x (in A) -> v.
    Distance is 3.
    Thus, removing any edge increases the diameter from 2 to 3.
    So every edge is essential.
    """
    n = 16
    edges = []
    
    # Iterate over all unique pairs
    for u in range(n):
        for v in range(u + 1, n):
            # Determine partition based on the 4th bit (index 3, value 8)
            # Part 0: 0..7, Part 1: 8..15
            if (u >> 3) != (v >> 3):
                edges.append([u, v])
                
    return edges
# EVOLVE-BLOCK-END
