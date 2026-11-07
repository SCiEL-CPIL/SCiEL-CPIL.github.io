"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a complete bipartite graph K_{8,8} on vertices 0..15.
    This graph has 64 edges, diameter 2, and every edge is critical
    for maintaining diameter 2 (removing any edge increases the diameter
    to 3 for the endpoints of that edge).
    """
    n = 16
    half = n // 2
    
    # Partitions:
    # Left: 0 .. half-1
    # Right: half .. n-1
    
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
