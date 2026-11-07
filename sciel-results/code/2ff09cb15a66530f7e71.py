"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    # Partition vertices into two sets of size 8
    # Set A: 0..7
    # Set B: 8..15
    # Connect every vertex in A to every vertex in B
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    
    # Ensure types are correct (int, list)
    # The loop above produces lists of ints, which is correct.
    return edges
# EVOLVE-BLOCK-END
