"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    # K_{8,8}: vertices 0-7 in part A, 8-15 in part B
    # Every edge is critical: removing (a,b) makes dist(a,b)=3, so diameter becomes 3
    edges = []
    for a in range(8):
        for b in range(8, 16):
            edges.append([a, b])
    return edges
# EVOLVE-BLOCK-END
