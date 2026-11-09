"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    """
    Constructs a complete bipartite graph K_{8,8}.
    This graph has:
    - 16 vertices
    - 64 edges
    - Diameter 2
    - Triangle-free. Removing any edge (u,v) increases the distance between u and v to 3
      (since there is no other path of length 2 between them in a bipartite graph with partitions of size > 1),
      thereby increasing the diameter to 3.
    """
    n = P['n']
    mid = n // 2
    edges = []
    for u in range(mid):
        for v in range(mid, n):
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
