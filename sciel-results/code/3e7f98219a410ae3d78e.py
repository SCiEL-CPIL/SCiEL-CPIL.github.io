"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    n = 15
    # Partition vertices into two sets A and B
    # To maximize edges in a bipartite graph K_{a,b} with a+b=n and a!=b,
    # we choose a and b as close as possible.
    # For n=15, a=7, b=8 gives 56 edges.
    # Degrees in A are 8, degrees in B are 7. 8 != 7, so constraint holds.
    a_size = 7
    b_size = 8
    
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
