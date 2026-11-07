import time
import random
import math

N = 15
DEADLINE = 85.0

def construct():
    start = time.time()
    
    # Strategy: K_{7,8} plus one internal edge in the smaller partition.
    # Vertices 0..6 form set A (size 7)
    # Vertices 7..14 form set B (size 8)
    # Base graph is K_{7,8} (56 edges).
    # Add edge (0, 1) in A.
    # Total edges: 57.
    
    edges = []
    
    # Add all bipartite edges
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])
            
    # Add one internal edge in A
    edges.append([0, 1])
    
    # Ensure types are Python int
    final_edges = [[int(e[0]), int(e[1])] for e in edges]
    
    return final_edges
