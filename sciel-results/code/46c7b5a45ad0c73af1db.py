"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Ten disjoint triangles: no four-cycle, independence number 10.
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    
    # This is a valid solution with n=30.
    # Any attempt to increase n requires a specific construction.
    # Without a verified 39-vertex graph, we return the base case.
    
    return {'n': 30, 'edges': edges}
# EVOLVE-BLOCK-END
