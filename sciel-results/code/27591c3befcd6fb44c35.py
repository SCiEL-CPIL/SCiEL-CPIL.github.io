import time

def construct():
    """
    Construct K_{8,8}: the unique extremal graph on 16 vertices with
    diameter exactly 2 and every edge critical (removing any edge
    increases diameter above 2 or disconnects the graph).
    
    By Mantel's theorem / extremal graph theory, floor(n^2/4) = 64 is
    the maximum number of edges in a triangle-free graph on 16 vertices,
    and K_{8,8} achieves this bound while satisfying all criticality
    constraints.
    """
    start_time = time.time()
    n = 16
    half = n // 2
    edges = []
    
    # Construct K_{8,8}: all edges between partition {0..7} and {8..15}
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    
    # Deadline safety guard
    if time.time() - start_time > 59:
        return edges
    
    # Ensure all elements are Python ints (not numpy types)
    edges = [[int(u), int(v)] for u, v in edges]
    
    return edges
