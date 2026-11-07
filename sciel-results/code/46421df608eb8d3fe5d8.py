"""Rewritten program following the research directive."""
# EVOLVE-BLOCK-START
def construct():
    edges = []
    # Construct a single K_{6,6} graph on vertices 0-11
    left = [0, 1, 2, 3, 4, 5]
    right = [6, 7, 8, 9, 10, 11]
    for u in left:
        for v in right:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
