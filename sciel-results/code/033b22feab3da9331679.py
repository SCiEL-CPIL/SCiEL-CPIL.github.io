"""
Constructs a diameter-2 graph on n=16 vertices with maximum edges such that
every edge is essential for maintaining diameter 2.
"""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a list of edges for a complete bipartite graph K_{8,8} on 16 vertices.
    
    Reasoning:
    1. The graph must have diameter 2.
    2. Deleting any edge must increase the diameter to > 2 or disconnect the graph.
    3. If an edge (u,v) has a common neighbor w, then deleting (u,v) leaves a path u-w-v of length 2.
       In dense graphs, this typically keeps the global diameter at 2, violating the condition.
       Thus, the graph must be triangle-free to ensure that for every edge (u,v), d(u,v) becomes >= 3 upon deletion.
    4. The maximum number of edges in a triangle-free graph on n vertices is floor(n^2/4) by Mantel's Theorem.
    5. For n=16, max edges = 16^2/4 = 64.
    6. The unique graph achieving this is K_{8,8}.
    7. K_{8,8} has diameter 2 and is triangle-free, so it satisfies all conditions.
    """
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    n = 16
    if n < 2:
        return []
    
    half = n // 2
    part1 = list(range(half))
    part2 = list(range(half, n))
    
    edges = []
    for u in part1:
        if time.time() > deadline:
            break
        for v in part2:
            edges.append([int(u), int(v)])
            
    return edges

# EVOLVE-BLOCK-END
