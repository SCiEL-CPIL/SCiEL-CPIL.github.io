"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    n = 16
    # Use complete bipartite graph K_{8,8}
    # This is triangle-free, has diameter 2, and is edge-maximal for triangle-free.
    # Removing any edge (u,v) between partitions leaves distance 3 between u and v
    # (since no path of length 2 exists between opposite partitions in bipartite graphs),
    # thus increasing diameter to 3.
    
    edges = []
    for u in range(n // 2):
        for v in range(n // 2, n):
            edges.append([u, v])
    
    # Verify time is still within deadline
    if time.time() > deadline:
        # Should not happen for this construction, but guard just in case
        pass
    
    return edges
# EVOLVE-BLOCK-END
