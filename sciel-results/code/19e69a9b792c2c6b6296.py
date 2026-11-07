# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 60  # 60s limit, keep margin

    half = n // 2
    edges = []
    
    # K_{8,8} is the unique maximum triangle-free graph on 16 vertices.
    # Partitions: A = {0..7}, B = {8..15}
    # Edges are all pairs (u, v) with u in A and v in B.
    for u in range(half):
        for v in range(half, n):
            if time.time() > deadline - 1:
                return edges
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
