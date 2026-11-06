"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Construct a graph with 8 disjoint copies of K4.
    # n = 32
    # Each K4 has 4 vertices and 6 edges.
    # Total edges = 8 * 6 = 48.
    # No 5-cycles: K4 is C5-free, disjoint union is C5-free.
    # Independence number: alpha(K4) = 1. Total alpha = 8 * 1 = 8.
    # 8 < 9, so no independent set of size 9.
    # This is the best known construction for R(C5, K9) > 32.
    
    n = 32
    edges = []
    
    for b in range(8):
        if time.time() > deadline:
            break
        base = 4 * b
        # Add all edges within the block of 4 vertices
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
