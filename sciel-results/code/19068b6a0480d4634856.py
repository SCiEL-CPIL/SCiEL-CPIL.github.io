"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    # To maximize edges while ensuring edge-essentiality, the graph must be triangle-free.
    # The maximum number of edges in a triangle-free graph on n vertices is floor(n^2/4),
    # achieved by the complete bipartite graph K_{n/2, n/2}.
    # For n=16, K_{8,8} has 64 edges and diameter 2.
    
    # Partition vertices into two sets of size 8
    set_a = list(range(8))
    set_b = list(range(8, 16))
    
    edges = []
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
