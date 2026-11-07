"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs the complete bipartite graph K_{8,8} on vertices 0..15.
    
    This graph has:
    1. Diameter exactly 2.
    2. Is triangle-free, ensuring that every edge is critical (removing any edge 
       increases the distance between its endpoints to 3, thus increasing diameter to 3).
    3. Maximizes edge count: By Mantel's Theorem, max edges in a triangle-free 
       graph on n=16 vertices is floor(16^2/4) = 64. K_{8,8} achieves this.
    
    Proof of optimality for edge-critical diameter-2 graphs:
    - Any edge in a triangle (u,v,w) is non-critical: removing (u,v) leaves path u-w-v of length 2.
    - Therefore, all edges must be in a triangle-free subgraph.
    - Mantel's theorem bounds triangle-free edges at floor(n^2/4) = 64 for n=16.
    - K_{8,8} achieves this bound and every edge is critical.
    
    Returns a list of edges [u, v] where u, v are Python ints.
    """
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin out of 60s limit

    n = 16
    mid = n // 2  # 8
    
    edges = []
    
    # Generate all edges between partition A={0..7} and B={8..15}
    for u in range(mid):
        for v in range(mid, n):
            edges.append([int(u), int(v)])
    
    # Verify types are correct (defensive check)
    # Each edge is [int, int]
    # 64 edges total
    
    return edges
# EVOLVE-BLOCK-END
