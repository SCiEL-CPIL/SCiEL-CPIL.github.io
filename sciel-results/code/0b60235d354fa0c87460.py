"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 16
    # Construct G via its complement G_bar.
    # For all edges in G to be essential, G must be triangle-free.
    # Max edges in triangle-free graph on n vertices is n^2/4 (Turan).
    # This is achieved by K_{n/2, n/2}, whose complement is K_{n/2} U K_{n/2}.
    # We define the complement cliques explicitly.
    
    size = n // 2
    clique1 = list(range(size))
    clique2 = list(range(size, n))
    
    edges = []
    # Edges in G are all pairs (u, v) where u and v are in different cliques of G_bar
    for u in clique1:
        for v in clique2:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
