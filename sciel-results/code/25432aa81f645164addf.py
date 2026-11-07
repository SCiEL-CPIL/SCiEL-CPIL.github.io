import time
import sys

# EVOLVE-BLOCK-START
def construct():
    n = 15
    # Partition A: 0..6, Partition B: 7..14
    A = list(range(0, 7))
    B = list(range(7, 15))
    
    edges = []
    
    # Add all edges between A and B (Complete Bipartite K_{7,8})
    for u in A:
        for v in B:
            edges.append([u, v])
            
    # Add one edge within A to increase count to 57
    # Edge (0, 1)
    edges.append([0, 1])
    
    # Verification logic (optional but good for debugging/safety)
    # In production, we trust the construction.
    
    return edges
# EVOLVE-BLOCK-END
