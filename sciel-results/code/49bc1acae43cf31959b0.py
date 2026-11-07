"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a maximal edge-critical diameter-2 graph on 16 vertices.
    Based on Mantel's Theorem, the maximum edges in a triangle-free graph on 16 vertices is 64.
    The complete bipartite graph K_{8,8} is triangle-free, has diameter 2,
    and is edge-critical: removing any edge (u,v) with u in A, v in B
    leaves no path of length 2 between u and v (since A and B are disjoint and
    no other common neighbors exist in the bipartite structure),
    thus increasing the distance to 3 and the diameter to 3.
    """
    n = 16
    half = n // 2
    
    # Partition vertices into two sets A and B
    set_a = set(range(half))
    set_b = set(range(half, n))
    
    edges = []
    # Generate all edges between A and B
    # Using list comprehension for efficiency and distinct structure
    for u in sorted(set_a):
        for v in sorted(set_b):
            edges.append([u, v])
            
    # Ensure types are strictly Python int and list
    # The loop above already produces Python ints and lists
    return edges
# EVOLVE-BLOCK-END
