"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    """Return edges of K(8,8) on vertices 0..15.
    
    K(8,8) is triangle-free with diameter 2. In a triangle-free graph,
    no two adjacent vertices share a common neighbor, so removing any
    edge (u,v) eliminates the only path of length <= 2 between u and v,
    making every edge essential for diameter 2.
    
    By Mantel's theorem, 64 edges is the maximum for a triangle-free
    graph on 16 vertices, so this is optimal.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    n = P['n']
    half = n // 2  # 8
    
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    
    # Verify types are correct - ensure pure Python ints
    result = []
    for e in edges:
        u, v = e
        if not isinstance(u, int) or isinstance(u, bool):
            u = int(u)
        if not isinstance(v, int) or isinstance(v, bool):
            v = int(v)
        result.append([u, v])
    
    # Deadline guard - return immediately as this is O(1) construction
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
