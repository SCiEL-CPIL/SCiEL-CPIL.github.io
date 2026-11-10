import time

def construct():
    n = 15
    # Partition vertices into two sets A and B
    # A = 0..6 (7 vertices)
    # B = 7..14 (8 vertices)
    edges = []
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])
    return edges
