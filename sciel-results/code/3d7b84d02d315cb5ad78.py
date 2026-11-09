"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns the edge list for the complete bipartite graph K_{8,8} on vertices 0..15.
    
    Properties:
    - Simple undirected graph with 64 edges.
    - Diameter exactly 2: any two vertices in the same partition have distance 2 (via any vertex in the other partition), and any two vertices in different partitions have distance 1.
    - Edge-critical for diameter: removing any edge (u,v) where u in A, v in B means that the distance between u and v becomes 3 (u -> other B -> other A -> v), so diameter becomes 3 > 2.
    - This is the maximum possible number of edges in a triangle-free graph on 16 vertices (Turán's theorem bound: floor(16^2/4) = 64).
    """
    n = 16
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
