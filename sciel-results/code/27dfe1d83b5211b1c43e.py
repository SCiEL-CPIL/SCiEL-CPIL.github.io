# EVOLVE-BLOCK-START
def construct():
    n = 16
    A = list(range(0, 8))
    B = list(range(8, 16))
    edges = []
    for a in A:
        for b in B:
            edges.append([a, b])
    return edges
# EVOLVE-BLOCK-END
