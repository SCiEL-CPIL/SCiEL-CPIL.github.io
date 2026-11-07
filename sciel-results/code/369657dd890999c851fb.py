import time
import itertools

def construct():
    # Deadline guard: 50 seconds to be safe within the 60s limit.
    start_time = time.time()
    deadline = start_time + 50.0

    # The Fano plane is the unique (up to isomorphism) minimal 3-uniform
    # 3-chromatic hypergraph. It has 7 edges.
    # Any 3-uniform hypergraph with fewer than 7 edges is 2-colorable.
    # Therefore, 7 is the global minimum edge count for this task.
    #
    # Vertices: 0, 1, 2, 3, 4, 5, 6 (Fano plane core)
    # Vertices: 7, 8 (isolated)
    #
    # Edges (lines of the Fano plane):
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]

    # A valid 3-coloring for the Fano plane.
    # The Fano plane is not 2-colorable, so it requires 3 colors.
    # Coloring:
    # v0: 0
    # v1: 1
    # v2: 2
    # v3: 1
    # v4: 2
    # v5: 0
    # v6: 1
    # v7: 0 (isolated)
    # v8: 0 (isolated)
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]

    # Ensure types are strict Python int and list
    edges_out = [[int(x) for x in e] for e in edges]
    colors_out = [int(c) for c in colors]

    # Return the optimal solution
    return {'edges': edges_out, 'colors': colors_out}
