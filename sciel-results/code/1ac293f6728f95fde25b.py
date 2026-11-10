"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 16}

def construct():
    n = P['n']
    # The problem requires a graph where every edge is critical for maintaining diameter 2.
    # This implies the graph must be triangle-free, as any edge in a triangle
    # would have an alternative path of length 2, potentially keeping the diameter at 2
    # even after removal (unless the graph is very sparse, but we want max edges).
    # Mantel's Theorem states the maximum number of edges in a triangle-free graph
    # on n vertices is floor(n^2 / 4), achieved by the complete bipartite graph K_{n/2, n/2}.
    # For n=16, this is K_{8,8} with 64 edges.
    
    mid = n // 2
    edges = []
    # Generate edges between partition A (0..mid-1) and Partition B (mid..n-1)
    for u in range(mid):
        for v in range(mid, n):
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
