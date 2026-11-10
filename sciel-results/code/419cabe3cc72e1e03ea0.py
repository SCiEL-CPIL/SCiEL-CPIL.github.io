"""Improved program for dense edge-critical diameter 2 graphs."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns the edges of the complete bipartite graph K_{8,8}.
    
    Reasoning:
    1. A graph is minimally diameter 2 (edge-critical) if and only if it has diameter 2 and contains no triangles. 
       - If an edge (u,v) is in a triangle (u,v,w), removing (u,v) leaves path u-w-v of length 2, so diameter remains 2. Thus (u,v) is not critical.
       - If the graph is triangle-free and has diameter 2, removing any edge (u,v) means there is no path of length 2 between u and v (as that would form a triangle). Since the graph is connected and diameter was 2, the new distance must be at least 3 (or infinity if disconnected, though K_{n,n} stays connected). Thus diameter increases.
    2. We want to maximize the number of edges in a triangle-free graph on 16 vertices.
    3. By Turan's Theorem, the maximum number of edges in a triangle-free graph on n vertices is floor(n^2 / 4).
    4. For n=16, max edges = 16^2 / 4 = 64. This is achieved by the complete bipartite graph K_{8,8}.
    5. K_{8,8} has diameter 2.
    6. Therefore, K_{8,8} is the unique graph (up to isomorphism) with the maximum number of edges that is minimally diameter 2.
    """
    edges = []
    # Partition vertices into two sets of size 8: {0, ..., 7} and {8, ..., 15}
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
