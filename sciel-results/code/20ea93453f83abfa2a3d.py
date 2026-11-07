"""Rewritten program following the research directive."""
# EVOLVE-BLOCK-START
def construct():
    n = 12
    # Partition vertices into two sets of 6
    # Set A: 0 to 5
    # Set B: 6 to 11
    set_a = list(range(0, 6))
    set_b = list(range(6, 12))
    
    edges = []
    # Complete bipartite graph K_6,6
    # Connect every vertex in A to every vertex in B
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
