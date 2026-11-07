"""
Constructs a graph on 16 vertices with diameter 2 such that removing any edge
increases the diameter above 2 or disconnects the graph.

The optimal structure is the complete bipartite graph K_{8,8} with 64 edges.
This is triangle-free, which is necessary: any triangle would make its edges
non-critical (removing one edge of a triangle leaves the other two, so distance
stays <= 2). The Mantel bound gives max edges in a triangle-free graph on 16
vertices as floor(16^2/4) = 64, achieved by K_{8,8}.

In K_{8,8}, removing edge (u,v) with u in A, v in B: u and v have no common
neighbor (u's neighbors are all of B, v's neighbors are all of A), so dist(u,v)
becomes 3, increasing the diameter above 2.
"""
import time

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin

    # Construct K_{8,8}: vertices 0-7 in partition A, 8-15 in partition B
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([int(u), int(v)])

    # K_{8,8} is the Mantel-optimal triangle-free graph on 16 vertices.
    # Every edge is critical: removing (u,v) makes dist(u,v)=3 since u and v
    # have no common neighbor in a complete bipartite graph.

    return edges
