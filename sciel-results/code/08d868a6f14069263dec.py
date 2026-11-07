"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs the complete bipartite graph K_{8,8} on vertices 0..15.
    
    This graph has 64 edges, which is the maximum possible for a triangle-free graph
    on 16 vertices (Mantel's Theorem). Since the graph must be triangle-free to satisfy
    the condition that removing any edge increases the diameter (a triangle would allow
    a path of length 2 to persist after edge removal), K_{8,8} is the unique optimal solution.
    
    Diameter is 2. Removing any edge (u, v) with u in A and v in B increases dist(u, v) to 3,
    thus increasing the diameter to 3.
    """
    n = 16
    half = n // 2
    edges = []
    
    # Partition A: 0..7, Partition B: 8..15
    for u in range(half):
        for v in range(half, n):
            # Ensure strict Python int types
            edges.append([int(u), int(v)])
            
    return edges

# EVOLVE-BLOCK-END
