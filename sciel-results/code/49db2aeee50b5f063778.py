import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before the 90s kill

    n = 16
    half = n // 2
    partition_a = list(range(half))
    partition_b = list(range(half, n))
    
    # Build K_{8,8}: complete bipartite graph between the two halves.
    # This is the extremal triangle-free graph on 16 vertices with 64 edges.
    # A diameter-2 edge-critical graph must be triangle-free (otherwise a
    # triangle's third edge would be redundant for the diameter-2 property).
    # Therefore 64 is the theoretical maximum, and every edge in K_{8,8}
    # is essential: deleting any edge (u,v) makes u and v at distance 3
    # (u -> a -> b -> v for any a in A\{u}, b in B\{v}), or disconnects
    # them in the limit, raising diameter above 2.
    
    edges = [[u, v] for u in partition_a for v in partition_b]
    
    return edges
