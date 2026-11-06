"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds margin from 150s limit
    
    # Current best known: 8 disjoint K4s
    # This graph has n=32.
    # It is C5-free (K4 has no C5, and disjoint union of C5-free graphs is C5-free).
    # Independence number is 8 (max 1 from each K4), which is < 9.
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    # Return the known valid construction
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
