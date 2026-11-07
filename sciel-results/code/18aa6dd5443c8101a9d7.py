"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 15
    # Part A: vertices 0..6 (7 vertices), Part B: vertices 7..14 (8 vertices)
    edges = []
    
    # All cross edges between A and B
    for a in range(7):
        for b in range(7, 15):
            edges.append([a, b])
    
    # A is a clique
    for i in range(7):
        for j in range(i+1, 7):
            edges.append([i, j])
    
    # B is K8 minus a perfect matching (4 edges removed)
    # Perfect matching in B: (7,8), (9,10), (11,12), (13,14)
    matching = {(7,8), (9,10), (11,12), (13,14)}
    for i in range(7, 15):
        for j in range(i+1, 15):
            if (i, j) not in matching:
                edges.append([i, j])
    
    # Verify: A vertices deg = 6 (A-internal) + 8 (B) = 14
    # B vertices deg = 7 (A) + 6 (B-internal) = 13
    # All 3-paths have endpoints in different parts, deg 14 vs 13, valid.
    
    # Convert to proper types
    result = []
    for e in edges:
        u = int(e[0])
        v = int(e[1])
        result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
