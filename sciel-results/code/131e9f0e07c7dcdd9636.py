import time


def construct():
    """
    Constructs a complete bipartite graph K_{7,8} on vertices 0..14.
    Partition A: {0, 1, 2, 3, 4, 5, 6} (7 vertices), degree 8 each.
    Partition B: {7, 8, 9, 10, 11, 12, 13, 14} (8 vertices), degree 7 each.

    In a bipartite graph, every 3-path s-u-v-t has s and t in different partitions.
    Since all A-vertices have degree 8 and all B-vertices have degree 7,
    deg(s) != deg(t) for any 3-path. Total edges = 56.
    """
    start_time = time.time()
    deadline = start_time + 85.0

    n = 15
    edges = []
    for u in range(7):
        if time.time() > deadline:
            break
        for v in range(7, 15):
            if time.time() > deadline:
                break
            edges.append([u, v])

    final_edges = []
    for e in edges:
        if isinstance(e[0], int) and not isinstance(e[0], bool) and isinstance(e[1], int) and not isinstance(e[1], bool):
            final_edges.append([int(e[0]), int(e[1])])

    return final_edges
