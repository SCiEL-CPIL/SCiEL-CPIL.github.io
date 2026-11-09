# EVOLVE-BLOCK-START
def construct():
    """
    Returns K_8,8: the complete bipartite graph on 16 vertices.
    
    Properties:
    - Triangle-free (no triangles)
    - Diameter exactly 2
    - Every edge is critical: removing any edge (a,b) with a in A, b in B
      makes dist(a,b) = 3 (no common neighbor exists after removal),
      so diameter increases to 3 > 2.
    - 64 edges, which is the Mantel bound (max edges in triangle-free graph
      on 16 vertices), so this is optimal.
    """
    A = list(range(0, 8))
    B = list(range(8, 16))
    edges = []
    for a in A:
        for b in B:
            edges.append([a, b])
    return edges
# EVOLVE-BLOCK-END
