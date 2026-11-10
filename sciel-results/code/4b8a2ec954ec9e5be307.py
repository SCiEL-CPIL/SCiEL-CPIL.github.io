import time

def construct():
    """
    Construct a graph on 16 vertices (0..15) with diameter exactly 2,
    where deleting any single edge increases the diameter above 2 or disconnects the graph.
    Maximize the number of edges.

    Reasoning:
    1. For an edge uv to be critical (removing it increases diameter > 2), u and v must have no common neighbor in the original graph. If they had a common neighbor w, the path u-w-v (length 2) would persist after removing uv, keeping d(u,v)=2. More rigorously, diameter-2 critical graphs must be triangle-free.
    2. If the graph is triangle-free, removing any edge uv ensures d(u,v) >= 3 (or infinity) in the remaining graph, thus increasing the diameter (which was 2) to at least 3.
    3. Therefore, the problem reduces to finding the maximum number of edges in a triangle-free graph on 16 vertices with diameter 2.
    4. By Turan's theorem, the maximum number of edges in a triangle-free graph on n vertices is floor(n^2/4). For n=16, this is 64, achieved by the complete bipartite graph K_{8,8}.
    5. K_{8,8} has diameter 2. Thus, K_{8,8} is the optimal solution and yields 64 edges.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit

    edges = []
    for u in range(8):
        if time.time() > deadline:
            # In this specific O(64) construction, this should never trigger,
            # but it satisfies the directive to have a deadline guard.
            return edges
        for v in range(8, 16):
            edges.append([u, v])
            
    return edges
