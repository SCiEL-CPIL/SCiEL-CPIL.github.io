import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns edges of a complete bipartite graph K_{7,8} on 15 vertices.
    
    Validity proof:
    - Part A = {0..6} (7 vertices), all degree 8
    - Part B = {7..14} (8 vertices), all degree 7
    - Any 3-edge path s-u-v-t must have u,v in opposite parts (no intra-part edges)
    - If u∈A, v∈B: s∈B (deg 7), t∈A (deg 8) → 7≠8 ✓
    - If u∈B, v∈A: s∈A (deg 8), t∈B (deg 7) → 8≠7 ✓
    - No path can have both endpoints in same part
    
    Edge count: 7 × 8 = 56. Score: 56/14 = 4.0.
    
    Optimality argument: Any improvement requires adding intra-part edges,
    which forces the other part's degrees to be all distinct. Making 8 degrees
    distinct requires removing ≥6 edges from the bipartite core, netting at
    most 56+1-6=51 < 56 for one added edge. Multiple intra-part edges don't
    help since the distinct-degree constraint is the same.
    """
    n = 15
    A = list(range(0, 7))
    B = list(range(7, 15))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
