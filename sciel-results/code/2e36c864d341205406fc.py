"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 15
    # Partition vertices into two sets A and B
    # A = {0, ..., 6} (size 7)
    # B = {7, ..., 14} (size 8)
    # The complete bipartite graph K_{7,8} has 56 edges.
    # Degrees in A are 8, degrees in B are 7.
    # Any path of length 3 s-u-v-t must have s and t in different partitions.
    # Thus deg(s) != deg(t) is always 8 != 7 or 7 != 8.
    
    edges = []
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
