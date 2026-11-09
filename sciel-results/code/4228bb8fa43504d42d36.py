import time

def construct():
    # The complete tripartite graph K6,6,6 is K4-free.
    # Its largest triangle-free induced subgraph consists of two entire parts, size 12.
    # We verify this is a valid starting point.
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    edges.append([u, v])
    return edges
