"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin below 90s wall-clock limit

    n = 16
    # By Mantel's theorem, the maximum number of edges in a triangle-free graph
    # on n vertices is floor(n^2/4). For n=16, this is 64, achieved by K_{8,8}.
    #
    # Key insight: For EVERY edge to be essential (deleting it must raise diameter
    # above 2 or disconnect the graph), the graph must be triangle-free.
    # Proof: If (u,v) lies in a triangle u-w-v-u, then deleting (u,v) still leaves
    # the path u-w-v of length 2, so dist(u,v) remains 2 — the edge is not essential.
    # Conversely, in a triangle-free graph, deleting edge (u,v) means u and v share
    # no common neighbor, so no length-2 path exists between them; their distance
    # becomes >2 (or infinite if disconnected).
    #
    # K_{8,8} has diameter exactly 2:
    #   - Vertices in the same partition have distance 2 (via any vertex in the other).
    #   - Vertices in different partitions have distance 1.
    #
    # 64 edges is the proven maximum.

    if time.time() - start_time > deadline:
        return []

    set_a = list(range(8))
    set_b = list(range(8, 16))

    edges = []
    for u in set_a:
        for v in set_b:
            edges.append([int(u), int(v)])

    return edges
# EVOLVE-BLOCK-END
