"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns the edges of K_{8,8}, the unique triangle-free graph on 16 vertices
    with maximum edge count (64) and diameter exactly 2.
    
    Every edge is critical: since the graph is triangle-free, removing any edge
    (u,v) eliminates the only path of length 1 between u and v. In K_{8,8}, 
    u and v are in different partitions, so no common neighbor exists (a common 
    neighbor would need to be adjacent to both, implying it is in both partitions, 
    which is impossible). Thus, the shortest path length becomes 3, increasing 
    the diameter from 2 to 3.
    
    By Mantel's Theorem, no triangle-free graph on 16 vertices can have more 
    than 64 edges, so this is optimal.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    n = 16
    half = n // 2

    # Generate edges of K_{8,8}
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])

    # The construction is deterministic and mathematically proven to be optimal.
    # No runtime verification is needed to avoid potential timeouts or errors.
    # Return immediately to ensure fast execution.
    return edges

# EVOLVE-BLOCK-END
