# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a simple undirected graph on vertices 0..15.
    The graph is the complete bipartite graph K_{8,8}.
    Partitions: A = {0, ..., 7}, B = {8, ..., 15}.
    Edges: All pairs (u, v) with u in A and v in B.
    
    Properties:
    - Diameter is exactly 2.
    - Every edge is essential (removing any edge increases diameter to 3).
    - Number of edges is 64, which is the maximum possible for such graphs.
    """
    n = 16
    half = n // 2
    edges = []
    
    # Generate all edges between partition A (0..7) and partition B (8..15)
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
