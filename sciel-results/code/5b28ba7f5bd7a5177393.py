"""Q3-free graph on 16 vertices: maximize edges."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # K8,8: bipartite graph with parts {0..7} and {8..15}
    # This is Q3-free because any Q3 embedded in it would need 4 vertices on each side,
    # inducing K4,4 which is 4-regular, not 3-regular.
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])

    return edges
