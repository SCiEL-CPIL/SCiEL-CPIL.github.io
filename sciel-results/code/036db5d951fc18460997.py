import time
import random

def construct():
    start_time = time.time()
    deadline = 90.0

    # We start from the known optimal solution: K(8,8) with 64 edges.
    # By Mantel's theorem, any triangle-free graph on 16 vertices has at most
    # floor(16^2/4) = 64 edges.
    #
    # The task requires:
    # 1. Diameter exactly 2.
    # 2. Every edge is critical (deleting it increases diameter > 2 or disconnects).
    #
    # If a graph has a critical edge (u,v), then dist(u,v) must become > 2 when
    # the edge is removed. This implies u and v share no common neighbor in
    # G \ {(u,v)}, i.e., no w such that w~u and w~v.
    # If there were a triangle (u,v,w) in G, then removing edge (u,v) would
    # still leave path u-w-v, so dist(u,v) would remain 2. Thus (u,v) would
    # not be critical.
    # Therefore, a graph where EVERY edge is critical must be triangle-free.
    #
    # Hence, any valid graph must be triangle-free and have at most 64 edges.
    # K(8,8) achieves 64 edges and satisfies all constraints, so it is optimal.

    # Construct K(8,8)
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([int(u), int(v)])

    # Return immediately; no need for further search since we've hit the theoretical maximum.
    return edges

BET: {mechanism: "K(8,8) achieves the Mantel bound of 64 edges, the theoretical maximum for triangle-free graphs on 16 vertices; since edge criticality forces triangle-freeness, no valid graph can exceed this, so 64 is optimal and this program is provably maximal.", q: 1.0, delta: [0, 0]}
