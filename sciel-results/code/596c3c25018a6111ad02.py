"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 40
    set_a = list(range(20))
    set_b = list(range(20, 40))
    
    # Strategy: Star construction in Set A to minimize book size.
    # We remove R bipartite edges incident to vertex 0 in A.
    # We add K intra-edges incident to vertex 0 in A.
    # Constraint: K - R = 1 (to maintain 401 edges from K_20,20).
    # To minimize the book size, we maximize R, which minimizes the degree of 0 to B.
    # Max K is 19 (connecting 0 to all other vertices in A).
    # Therefore, max R is 18.
    # This results in |N(0) intersect B| = 20 - 18 = 2.
    # The book size for any edge (0, i) is |N(0) intersect N(i)| = 2.
    # No other intra-edges exist, so max book size is 2.
    
    R = 18
    K = 19
    
    # Intra-edges: (0, i) for i in 1..19
    # Removed bipartite edges: (0, 20+j) for j in 0..17
    
    removed_edges = set()
    for j in range(R):
        removed_edges.add((0, 20 + j))
        
    edges = []
    
    # Add bipartite edges, skipping removed ones
    for u in set_a:
        for v in set_b:
            if (u, v) not in removed_edges:
                edges.append([u, v])
                
    # Add intra-edges
    for i in range(1, K + 1):
        edges.append([0, i])
        
    return edges
# EVOLVE-BLOCK-END
